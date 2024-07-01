import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import math

from utils.uflow_utils import flow_to_warp, resample2, compute_range_map, mask_invalid, census_loss_no_penalty, image_grads, robust_l1, abs_robust_loss, upsample, ssim_loss, charbonnier
from utils.triag_solve import BackwardSubst, inverse_l1norm, matrix_vector_product, matrix_vector_product_T, NaturalGradientIdentityT, NaturalGradientIdentityC
from utils.misc_utils import gaussian_mixture_log_pdf


def data_loss_no_penalty(im1_0, im2_0, flow12_2, flow21_2, align_corners, occu_mean, data_loss, mean12_2=None, mean21_2=None):
    """
    Computes the per-pixel data loss and weight map before applying penalty functions
    :param: im1_0: (batch, channels, height, width) tensor representing the first image at original resolution (level 0)
    :param: im2_0: (batch, channels, height, width) tensor representing the second mage at original resolution (level 0)
    :param: flow12_2: (batch*n_samples, 2, height, width) tensor representing the forward flow at 1/4 resolution (level 2)
    :param: flow21_2: (batch*n_samples, 2, height, width) tensor representing the backward flow at 1/4 resolution (level 2)
    :param: mean12_2: (batch, 2, height, width) tensor representing the forward flow at 1/4 resolution (level 2)
    :param: mean21_2: (batch, 2, height, width) tensor representing the backward flow at 1/4 resolution (level 2)
    :return: loss12: (batch, 1, height, width) forward loss,
             weight12: (batch, 1, height, width) forward weight,
    """

    # Upsample flow 4x #
    ####################
    flow12_0 = upsample(flow12_2, scale_factor=4, is_flow=True, align_corners=align_corners)
    flow21_0 = upsample(flow21_2, scale_factor=4, is_flow=True, align_corners=align_corners)

    # Warp the images #
    ###################
    warp12_0 = flow_to_warp(flow12_0)
    im1_recons = resample2(im2_0.detach(), warp12_0)

    # Calculate border and occlusion masks #
    ########################################
    if occ_type == 'mean':
        mean12_0 = upsample(mean12_2, scale_factor=4, is_flow=True, align_corners=align_corners)
        mean21_0 = upsample(mean21_2, scale_factor=4, is_flow=True, align_corners=align_corners)

        mean_warp12_0 = flow_to_warp(mean12_0)
        valid_mask1 = mask_invalid(mean_warp12_0)
        occu_mask1 = torch.clamp(compute_range_map(mean21_0), min=0., max=1.)
        mask1 = torch.detach(occu_mask1 * valid_mask1)

    elif occ_type == 'sample':
        valid_mask1 = mask_invalid(warp12_0)
        occu_mask1 = torch.clamp(compute_range_map(flow21_0), min=0., max=1.)
        mask1 = torch.detach(occu_mask1 * valid_mask1)

    elif occ_type == 'none':
        mask1 = torch.detach(mask_invalid(warp12_0))

    else:
        raise NotImplementedError('Occlusion type {} not implemented!'.format(occ_type))

    # Calculate photometric loss on level 0 #
    #########################################
    pixel_loss = []
    pixel_weight = []
    for type in data_loss:
        if type == "census":
            l, w = census_loss_no_penalty(im1_0, im1_recons, mask1)
        elif type == "ssim":
            l, w = ssim_loss(im1_0, im1_recons, mask1)

        pixel_loss.append(l)
        pixel_weight.append(w)

    return pixel_loss, pixel_weight


def smooth_loss_no_penalty(im1_0, flow12_2, align_corners, edge_constant, edge_asymp):
    # Calculate smoothness loss on level 2 #
    ############################################################
    _, _, height, width = im1_0.size()
    im1_1 = F.interpolate(im1_0, scale_factor=0.5, mode='bilinear', align_corners=align_corners)
    im1_2 = F.interpolate(im1_1, scale_factor=0.5, mode='bilinear', align_corners=align_corners)

    # Forward -----------
    im1_gx, im1_gy = image_grads(im1_2.detach())
    weights1_x = edge_asymp + (1.0-edge_asymp)*torch.exp(-torch.mean(torch.abs(edge_constant * im1_gx), 1, keepdim=True))
    weights1_y = edge_asymp + (1.0-edge_asymp)*torch.exp(-torch.mean(torch.abs(edge_constant * im1_gy), 1, keepdim=True))

    flow12_x, flow12_y = image_grads(flow12_2)
    weight12_x = weights1_x / 2.
    weight12_y = weights1_y / 2.

    return flow12_x, weight12_x, flow12_y, weight12_y


def log_gmm(x, pi, beta):
    pi = torch.tensor(pi, device=x.device)
    beta = torch.tensor(beta, device=x.device)
    arg = -beta * torch.square(x).unsqueeze(-1) / 2
    w = pi * torch.sqrt(beta) / math.sqrt(2 * torch.pi)
    c = torch.max(arg, dim=-1).values
    return c + torch.log(torch.sum(w * torch.exp(arg - c.unsqueeze(-1)), dim=-1))


class SlowDownIdentity(Function):
    @staticmethod
    def forward(ctx, A, B, C, X):
        return A, B, C, X

    @staticmethod
    @once_differentiable
    def backward(ctx, dA, dB, dC, dX):
        return dA / 100.0, dB / 100.0, dC / 100.0, dX


class UFlowElboLoss(nn.modules.Module):
    def __init__(self, cfg):
        super(UFlowElboLoss, self).__init__()
        self.cfg = cfg

        self.Normal = torch.distributions.Normal(0, 1)
        if torch.cuda.is_available():
            self.Normal.loc = self.Normal.loc.cuda() # hack to get sampling on the GPU
            self.Normal.scale = self.Normal.scale.cuda()

    def reparam_diag(self, mean, log_diag, nsamples=1):
        """
        Generates normal distributed samples with a given mean and variance.
        mean: mean - tensor of size (batch, 2, height, width)
        log_diag: log(variance)/2=log(std) - tensor of size (batch, 2, height, width)
        returns: samples - tensor of size (Nsamples*batch, 2, height, width)
        """
        mean = mean.repeat(nsamples, 1, 1, 1)
        log_diag = log_diag.repeat(nsamples, 1, 1, 1)
        z = mean + torch.exp(log_diag) * self.Normal.sample(mean.size())
        return z

    def reparam_diag_inv(self, mean, log_diag, nsamples=1):
        """
        Generates normal distributed samples with a given mean and precision.
        mean: mean - tensor of size (batch, 2, height, width)
        log_diag: log(precision)/2 - tensor of size (batch, 2, height, width)
        returns: samples - tensor of size (Nsamples*batch, 2, height, width)
        """
        mean = mean.repeat(nsamples, 1, 1, 1)
        log_diag = log_diag.repeat(nsamples, 1, 1, 1)
        z = mean + torch.exp(-log_diag) * self.Normal.sample(mean.size())
        return z

    def reparam_triag(self, mean, diag, left, over, nsamples=1):
        mean = mean.repeat(nsamples, 1, 1, 1)
        diag = diag.repeat(nsamples, 1, 1, 1)
        left = left.repeat(nsamples, 1, 1, 1)
        over = over.repeat(nsamples, 1, 1, 1)
        eps = self.Normal.sample(mean.size())
        z = mean + matrix_vector_product(diag, left, over, eps)
        return z

    def reparam_triag_inv(self, mean, diag, left, over, nsamples=1):
        mean = mean.repeat(nsamples, 1, 1, 1)
        diag = diag.repeat(nsamples, 1, 1, 1)
        left = left.repeat(nsamples, 1, 1, 1)
        over = over.repeat(nsamples, 1, 1, 1)
        eps = self.Normal.sample(mean.size())
        z = mean + BackwardSubst.apply(diag, left, over, eps)
        return z

    def reparam_gmm(self, mean, std, weights, nsamples=1):
        """
        Generates gaussian mixture distributed samples with a given mean and variance.
        mean: mean - tensor of size (batch, 2*K, height, width)
        std: standard deviation - tensor of size (batch, 2, height, width)
        weights: weights - tensor of size(batch, K)
        returns: samples - tensor of size (nsamples*batch, 2, height, width)
        """
        rows, cols = mean.shape[2:]
        z = torch.multinomial(weights, num_samples=nsamples, replacement=True)      # (batch, nsamples)
        z = z[:, :, None, None].repeat(1, 1, rows, cols)                            # (batch, nsamples, rows, cols)
        # Batches change fast, samples slow (to be consistent with the diag array)
        mean_u = torch.gather(mean, 1, 2*z).transpose(1,0).reshape(-1, 1, rows, cols)    # (batch*nsamples, 1, rows, cols)
        std_u = torch.gather(std, 1, 2*z).transpose(1,0).reshape(-1, 1, rows, cols)      # (batch*nsamples, 1, rows, cols)
        mean_v = torch.gather(mean, 1, 2*z+1).transpose(1,0).reshape(-1, 1, rows, cols)  # (batch*nsamples, 1, rows, cols)
        std_v = torch.gather(std, 1, 2*z+1).transpose(1,0).reshape(-1, 1, rows, cols)      # (batch*nsamples, 1, rows, cols)
        mean = torch.cat((mean_u, mean_v), dim=1)
        std = torch.cat((std_u, std_v), dim=1)
        z = mean + std * self.Normal.sample(std.size())
        return z

    def forward(self, res_dict, im1_0, im2_0):
        """
        :param output: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :param target: image pairs Nx6xHxW
        :return:
        """

        # Get level 2 outputs - before upsampling
        if self.cfg.approx == 'diag':
            mean12_2 = res_dict['flows_fw'][2][:, 0:2]
            log_diag12_2 = res_dict['flows_fw'][2][:, 2:4]
            mean21_2 = res_dict['flows_bw'][2][:, 0:2]
            log_diag21_2 = res_dict['flows_bw'][2][:, 2:4]
            diag12_2 = torch.exp(log_diag12_2)
            diag21_2 = torch.exp(log_diag21_2)

        elif self.cfg.approx == 'sparse':
            mean12_2 = res_dict['flows_fw'][2][:, 0:2]
            log_diag12_2 = res_dict['flows_fw'][2][:, 2:4]
            left12_2 = res_dict['flows_fw'][2][:, 4:6, :, :-1]
            over12_2 = res_dict['flows_fw'][2][:, 6:8, :-1, :]

            mean21_2 = res_dict['flows_bw'][2][:, 0:2]
            log_diag21_2 = res_dict['flows_bw'][2][:, 2:4]
            left21_2 = res_dict['flows_bw'][2][:, 4:6, :, :-1]
            over21_2 = res_dict['flows_bw'][2][:, 6:8, :-1, :]

            # Ensure that the matrices are diagonally dominant
            diag12_2 = torch.exp(log_diag12_2)
            diag21_2 = torch.exp(log_diag21_2)
            if self.cfg.diag_dominant is not None:
                offdiag12 = F.pad(torch.abs(left12_2), (1, 0)) + F.pad(torch.abs(over12_2), (0, 0, 1, 0))
                diag12_2 = diag12_2 + self.cfg.diag_dominant * offdiag12
                offdiag21 = F.pad(torch.abs(left21_2), (1, 0)) + F.pad(torch.abs(over21_2), (0, 0, 1, 0))
                diag21_2 = diag21_2 + self.cfg.diag_dominant * offdiag21

                # Update the log values
                log_diag12_2 = torch.log(diag12_2)
                log_diag21_2 = torch.log(diag21_2)

        elif self.cfg.approx == 'mixture':
            K = self.cfg.n_components
            mean12_2 = res_dict['flows_fw'][2][:, 0:2*K]
            log_diag12_2 = res_dict['flows_fw'][2][:, 2*K:4*K]
            mean21_2 = res_dict['flows_bw'][2][:, 0:2*K]
            log_diag21_2 = res_dict['flows_bw'][2][:, 2*K:4*K]
            weights12 = torch.ones((mean12_2.size(0), K), device=mean12_2.device) / K
            weights21 = torch.ones((mean21_2.size(0), K), device=mean21_2.device) / K
            if 'weights_fw' in res_dict:
                weights12 = res_dict['weights_fw']
                weights21 = res_dict['weights_bw']

            diag12_2 = torch.exp(log_diag12_2)
            diag21_2 = torch.exp(log_diag21_2)
            #print("mean, var:", torch.mean(log_diag12_2), torch.std(log_diag12_2))
            #print("mean, var:", torch.mean(log_diag21_2), torch.std(log_diag21_2))


        # Apply identity function that implements natural gradient computation #
        ########################################################################
        if self.cfg.natural_grad:
            if self.cfg.inv_cov:
                diag12_2, left12_2, over12_2, mean12_2 = NaturalGradientIdentityT.apply(
                    diag12_2.contiguous(), left12_2.contiguous(), over12_2.contiguous(), mean12_2.contiguous()
                )
                diag21_2, left21_2, over21_2, mean21_2 = NaturalGradientIdentityT.apply(
                    diag21_2.contiguous(), left21_2.contiguous(), over21_2.contiguous(), mean21_2.contiguous()
                )
            else:
                diag12_2, left12_2, over12_2, mean12_2 = NaturalGradientIdentityC.apply(
                    diag12_2.contiguous(), left12_2.contiguous(), over12_2.contiguous(), mean12_2.contiguous()
                )
                diag21_2, left21_2, over21_2, mean21_2 = NaturalGradientIdentityC.apply(
                    diag21_2.contiguous(), left21_2.contiguous(), over21_2.contiguous(), mean21_2.contiguous()
                )

            # Update the log values
            log_diag12_2 = torch.log(diag12_2)
            log_diag21_2 = torch.log(diag21_2)

        if self.cfg.slow_down:
            diag12_2, left12_2, over12_2, mean12_2 = SlowDownIdentity.apply(diag12_2, left12_2, over12_2, mean12_2)
            diag21_2, left21_2, over21_2, mean21_2 = SlowDownIdentity.apply(diag21_2, left21_2, over21_2, mean21_2)
            # Update the log values
            log_diag12_2 = torch.log(diag12_2)
            log_diag21_2 = torch.log(diag21_2)

        # Regularization of the off-diagonal entries #
        ##############################################
        loss_offdiag = 0
        if self.cfg.approx == 'sparse':
            loss_offdiag = (torch.mean(torch.square(left12_2)) + torch.mean(torch.square(over12_2))) / 2.0
            if self.cfg.with_bk:
                loss_offdiag += (torch.mean(torch.square(left21_2)) + torch.mean(torch.square(over21_2))) / 2.0

        # Calculate l1 norm of the precision matrix inverse #
        #####################################################
        K, L = mean12_2.shape[0:2]
        inv_l1norm = 0.0
        # if not self.cfg.diag:
        #     for k in range(K):
        #         for l in range(L):
        #             out12 = inverse_l1norm(diag12_2[k, l], left12_2[k, l], over12_2[k, l], n_iter=10).item()
        #             out21 = inverse_l1norm(diag21_2[k, l], left21_2[k, l], over21_2[k, l], n_iter=10).item()
        #             inv_l1norm = max(inv_l1norm, out12, out21)

        # Reparametrization trick #
        ###########################
        if self.cfg.approx == 'diag' and not self.cfg.inv_cov:
            flow12_2 = self.reparam_diag(mean12_2, log_diag12_2, nsamples=self.cfg.n_samples)
            flow21_2 = self.reparam_diag(mean21_2, log_diag21_2, nsamples=self.cfg.n_samples)
        elif self.cfg.approx == 'diag' and self.cfg.inv_cov:
            flow12_2 = self.reparam_diag_inv(mean12_2, log_diag12_2, nsamples=self.cfg.n_samples)
            flow21_2 = self.reparam_diag_inv(mean21_2, log_diag21_2, nsamples=self.cfg.n_samples)
        elif self.cfg.approx == 'sparse' and not self.cfg.inv_cov:
            flow12_2 = self.reparam_triag(mean12_2, diag12_2, left12_2, over12_2, nsamples=self.cfg.n_samples)
            flow21_2 = self.reparam_triag(mean21_2, diag21_2, left21_2, over21_2, nsamples=self.cfg.n_samples)
        elif self.cfg.approx == 'sparse' and self.cfg.inv_cov:
            flow12_2 = self.reparam_triag_inv(mean12_2, diag12_2, left12_2, over12_2, nsamples=self.cfg.n_samples)
            flow21_2 = self.reparam_triag_inv(mean21_2, diag21_2, left21_2, over21_2, nsamples=self.cfg.n_samples)
        elif self.cfg.approx == 'mixture' and not self.cfg.inv_cov:
            flow12_2 = self.reparam_gmm(mean12_2, diag12_2, weights12, nsamples=self.cfg.n_samples)
            flow21_2 = self.reparam_gmm(mean21_2, diag21_2, weights21, nsamples=self.cfg.n_samples)
        elif self.cfg.approx == 'mixture' and self.cfg.inv_cov:
            raise NotImplementedError('Inverse covariance parametrization is not implemented for mixture variational approximation.')

        # Repeat to take into account number of MC samples #
        ####################################################
        im1_0 = im1_0.repeat(self.cfg.n_samples, 1, 1, 1)
        im2_0 = im2_0.repeat(self.cfg.n_samples, 1, 1, 1)

        # Calculate entropy loss #
        ##########################
        if self.cfg.approx == 'diag' and not self.cfg.inv_cov:
            if self.cfg.approx_entropy:
                tmp12 = (flow12_2 - mean12_2.detach()) / diag12_2.detach()
                loss_entropy = self.cfg.w_entropy * torch.sum(tmp12*tmp12/2, dim=1).mean()
                if self.cfg.with_bk:
                    tmp21 = (flow21_2 - mean21_2.detach()) / diag21_2.detach()
                    loss_entropy += self.cfg.w_entropy * torch.sum(tmp21*tmp21/2, dim=1).mean()
            else:
                loss_entropy = self.cfg.w_entropy * torch.sum(log_diag12_2, dim=1).mean()
                if self.cfg.with_bk:
                    loss_entropy += self.cfg.w_entropy * torch.sum(log_diag21_2, dim=1).mean()
        elif self.cfg.approx == 'diag' and self.cfg.inv_cov:
            loss_entropy = -self.cfg.w_entropy * torch.sum(log_diag12_2, dim=1).mean()
            if self.cfg.with_bk:
                loss_entropy -= self.cfg.w_entropy * torch.sum(log_diag21_2, dim=1).mean()
        elif self.cfg.approx == 'sparse' and not self.cfg.inv_cov:
            loss_entropy = self.cfg.w_entropy * torch.sum(log_diag12_2, dim=1).mean()
            if self.cfg.with_bk:
                loss_entropy += self.cfg.w_entropy * torch.sum(log_diag21_2, dim=1).mean()
        elif self.cfg.approx == 'sparse' and self.cfg.inv_cov:
            if self.cfg.approx_entropy:
                tmp12 = matrix_vector_product_T(diag12_2.detach(), left12_2.detach(), over12_2.detach(),
                                                flow12_2 - mean12_2.detach())
                loss_entropy = self.cfg.w_entropy * torch.sum(tmp12*tmp12/2, dim=1).mean()
                if self.cfg.with_bk:
                    tmp21 = matrix_vector_product_T(diag21_2.detach(), left21_2.detach(), over21_2.detach(),
                                                    flow21_2 - mean21_2.detach())
                    loss_entropy += self.cfg.w_entropy * torch.sum(tmp21*tmp21/2, dim=1).mean()
            else:
                loss_entropy = -self.cfg.w_entropy * torch.sum(log_diag12_2, dim=1).mean()
                if self.cfg.with_bk:
                    loss_entropy -= self.cfg.w_entropy * torch.sum(log_diag21_2, dim=1).mean()
        elif self.cfg.approx == 'mixture':
            loss_entropy = -self.cfg.w_entropy * gaussian_mixture_log_pdf(flow12_2, mean12_2, log_diag12_2, weights12).mean()
            if self.cfg.with_bk:
                loss_entropy -= self.cfg.w_entropy * gaussian_mixture_log_pdf(flow21_2, mean21_2, log_diag21_2, weights21).mean()

        # Data loss on level 0 #
        ########################
        penalty_list = []
        for type in self.cfg.data_penalty:
            if type == "identity":
                penalty_list.append(lambda x: x)
            elif type == "abs_robust_loss":
                penalty_list.append(abs_robust_loss)
            elif type == "gmm":
                penalty_list.append(lambda x: -log_gmm(x, self.cfg.penalty_census_pi, self.cfg.penalty_census_beta))
            elif type == "charbonnier":
                penalty_list.append(charbonnier)

        loss_warp = 0
        data_pixel_loss12, data_pixel_weight12, mask12 = data_loss_no_penalty(
            im1_0, im2_0, flow12_2, flow21_2, self.cfg.align_corners, self.cfg.occ_type, self.cfg.data_loss,
            mean12_2, mean21_2
        )
        for pixel_loss, pixel_weight, weight, penalty in zip(data_pixel_loss12, data_pixel_weight12, self.cfg.data_weight, penalty_list):
            loss_warp += torch.sum(pixel_weight * weight * penalty(pixel_loss))

        if self.cfg.with_bk:
            # Here the arguments are passed in reversed order!
            data_pixel_loss21, data_pixel_weight21, _ = data_loss_no_penalty(
                im2_0, im1_0, flow21_2, flow12_2, self.cfg.align_corners, self.cfg.occ_type, self.cfg.data_loss,
                mean21_2, mean12_2
            )
            for pixel_loss, pixel_weight, weight, penalty in zip(data_pixel_loss21, data_pixel_weight21, self.cfg.data_weight, penalty_list):
                loss_warp += torch.sum(pixel_weight * weight * penalty(pixel_loss))

        # Smoothness loss on level 2 #
        ##############################
        if self.cfg.penalty_smooth == "charbonnier":
            penalty_func_smooth = charbonnier
        elif self.cfg.penalty_smooth == "gmm":
            penalty_func_smooth = lambda x: -log_gmm(x, self.cfg.penalty_smooth_pi, self.cfg.penalty_smooth_beta)

        smooth_loss12_x, smooth_weight12_x, smooth_loss12_y, smooth_weight12_y = smooth_loss_no_penalty(
            im1_0, flow12_2, self.cfg.align_corners, self.cfg.edge_constant, self.cfg.edge_asymp
        )
        # In contrast to data loss, the smoothness loss is AVERAGED over pixels (comes from the UFlow code)
        loss_smooth = torch.mean(smooth_weight12_x * self.cfg.w_smooth * penalty_func_smooth(torch.mean(smooth_loss12_x**2, dim=1))) \
                      + torch.mean(smooth_weight12_y * self.cfg.w_smooth * penalty_func_smooth(torch.mean(smooth_loss12_y**2, dim=1)))
        if self.cfg.with_bk:
            # Here the arguments are passed in reversed order!
            smooth_loss21_x, smooth_weight21_x, smooth_loss21_y, smooth_weight21_y = smooth_loss_no_penalty(
                im2_0, flow21_2, self.cfg.align_corners, self.cfg.edge_constant, self.cfg.edge_asymp
            )
            loss_smooth += torch.mean(smooth_weight21_x * self.cfg.w_smooth * penalty_func_smooth(smooth_loss21_x)) \
                          + torch.mean(smooth_weight21_y * self.cfg.w_smooth * penalty_func_smooth(smooth_loss21_y))
            loss_smooth += torch.mean(smooth_weight21_x * self.cfg.w_smooth * penalty_func_smooth(torch.mean(smooth_loss21_x**2, dim=1))) \
                          + torch.mean(smooth_weight21_y * self.cfg.w_smooth * penalty_func_smooth(torch.mean(smooth_loss21_y**2, dim=1)))

        # Calculate total loss #
        ########################
        total_loss = loss_warp + loss_smooth - loss_entropy
        if self.cfg.approx == 'sparse':
            total_loss += self.cfg.offdiag_reg*loss_offdiag

        return total_loss, loss_warp, loss_smooth, loss_entropy, loss_offdiag
