import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.uflow_utils import flow_to_warp, resample2, compute_range_map, mask_invalid, census_loss, image_grads, robust_l1, upsample
from utils.triag_solve import BackwardSubst, inverse_l1norm


class UFlowElboLoss(nn.modules.Module):
    def __init__(self, cfg):
        super(UFlowElboLoss, self).__init__()
        self.cfg = cfg

        self.Normal = torch.distributions.Normal(0, 1)
        self.Normal.loc = self.Normal.loc.cuda() # hack to get sampling on the GPU
        self.Normal.scale = self.Normal.scale.cuda()

    def reparam_diag(self, mean, log_var, nsamples=1):
        """
        Generates normal distributed samples with a given mean and variance.
        mean: mean - tensor of size (batch, 2, height, width)
        log_var: log(variance) - tensor of size (batch, 2, height, width)
        returns: samples - tensor of size (Nsamples*batch, 2, height, width)
        """
        mean = mean.repeat(nsamples, 1, 1, 1)
        log_var = log_var.repeat(nsamples, 1, 1, 1)
        z = mean + torch.exp(log_var / 2.0) * self.Normal.sample(mean.size())
        return z

    def reparam_triag(self, mean, diag, left, over, nsamples=1):
        mean = mean.repeat(nsamples, 1, 1, 1)
        diag = diag.repeat(nsamples, 1, 1, 1)
        left = left.repeat(nsamples, 1, 1, 1)
        over = over.repeat(nsamples, 1, 1, 1)
        eps = self.Normal.sample(mean.size())
        z = mean + BackwardSubst.apply(diag, left, over, eps)
        return z

    def forward(self, output, target):
        """

        :param output: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :param target: image pairs Nx6xHxW
        :return:
        """

        # Get level 2 outputs - before upsampling
        if self.cfg.diag:
            mean12_2 = output[2][:, 0:2]
            log_var12_2 = output[2][:, 2:4]
            mean21_2 = output[2][:, 4:6]
            log_var21_2 = output[2][:, 6:8]
        else:
            mean12_2 = output[2][:, 0:2]
            log_diag12_2 = output[2][:, 2:4]
            left12_2 = output[2][:, 4:6, :, :-1]
            over12_2 = output[2][:, 6:8, :-1, :]

            mean21_2 = output[2][:, 8:10]
            log_diag21_2 = output[2][:, 10:12]
            left21_2 = output[2][:, 12:14, :, :-1]
            over21_2 = output[2][:, 14:16, :-1, :]

            # Ensure that the matrices are diagonally dominant
            diag12_2 = torch.exp(log_diag12_2)
            diag21_2 = torch.exp(log_diag21_2)
            if self.cfg.diag_dominant:
                diag12_2 = diag12_2 \
                           + torch.nn.functional.pad(torch.abs(left12_2), (1, 0)) \
                           + torch.nn.functional.pad(torch.abs(over12_2), (0, 0, 1, 0))

                diag21_2 = diag21_2 \
                           + torch.nn.functional.pad(torch.abs(left21_2), (1, 0)) \
                           + torch.nn.functional.pad(torch.abs(over21_2), (0, 0, 1, 0))

        im1_0 = target[:, :3]
        im2_0 = target[:, 3:]

        # Calculate l1 norm of the precision matrix inverse #
        #####################################################
        K, L = mean12_2.shape[0:2]
        inv_l1norm = 0.0
        for k in range(K):
            for l in range(L):
                out12 = inverse_l1norm(diag12_2[k, l], left12_2[k, l], over12_2[k, l]).item()
                out21 = inverse_l1norm(diag21_2[k, l], left21_2[k, l], over21_2[k, l]).item()
                inv_l1norm = max(inv_l1norm, out12, out21)

        # Calculate entropy loss #
        ##########################
        if self.cfg.diag:
            loss_entropy = self.cfg.w_entropy * torch.sum(log_var12_2, dim=1).mean() / 2.0
            if self.cfg.with_bk:
                loss_entropy = self.cfg.w_entropy * torch.sum(log_var21_2, dim=1).mean() / 2.0
        else:
            if self.cfg.diag_dominant:
                loss_entropy = -self.cfg.w_entropy * torch.sum(torch.log(diag12_2), dim=1).mean()
                if self.cfg.with_bk:
                    loss_entropy -= self.cfg.w_entropy * torch.sum(torch.log(diag21_2), dim=1).mean()
            else:
                loss_entropy = -self.cfg.w_entropy * torch.sum(log_diag12_2, dim=1).mean()
                if self.cfg.with_bk:
                    loss_entropy -= self.cfg.w_entropy * torch.sum(log_diag21_2, dim=1).mean()

        # Reparametrization trick #
        ###########################
        if self.cfg.diag:
            flow12_2 = self.reparam_diag(mean12_2, log_var12_2)
            flow21_2 = self.reparam_diag(mean21_2, log_var21_2)
        else:
            flow12_2 = self.reparam_triag(mean12_2, diag12_2, left12_2, over12_2)
            flow21_2 = self.reparam_triag(mean21_2, diag21_2, left21_2, over21_2)

        # Upsample flow 4x
        flow12_1 = upsample(flow12_2, is_flow=True, align_corners=self.cfg.align_corners)
        flow12_0 = upsample(flow12_1, is_flow=True, align_corners=self.cfg.align_corners)
        flow21_1 = upsample(flow21_2, is_flow=True, align_corners=self.cfg.align_corners)
        flow21_0 = upsample(flow21_1, is_flow=True, align_corners=self.cfg.align_corners)

        # Warp the images #
        ###################
        warp12_0 = flow_to_warp(flow12_0)
        im1_recons = resample2(im2_0.detach(), warp12_0)
        if self.cfg.with_bk:
            warp21_0 = flow_to_warp(flow21_0)
            im2_recons = resample2(im1_0.detach(), warp21_0)

        # Calculate border and occlusion masks #
        ########################################
        valid_mask1 = mask_invalid(warp12_0)
        occu_mask1 = torch.clamp(compute_range_map(flow21_0), min=0., max=1.)
        mask1 = torch.detach(occu_mask1 * valid_mask1)
        if self.cfg.with_bk:
            valid_mask2 = mask_invalid(warp21_0)
            occu_mask2 = torch.clamp(compute_range_map(flow12_0), min=0., max=1.)
            mask2 = torch.detach(occu_mask2 * valid_mask2)

        # Calculate photometric loss on level 0 #
        ############################################################
        loss_warp = self.cfg.w_census * census_loss(im1_0, im1_recons, mask1)
        if self.cfg.with_bk:
            loss_warp += self.cfg.w_census * census_loss(im2_0, im2_recons, mask2)

        # Calculate smoothness loss on level 2 #
        ############################################################
        _, _, height, width = im1_0.size()
        im1_1 = F.interpolate(im1_0, scale_factor=0.5, mode='bilinear', align_corners=self.cfg.align_corners)
        im1_2 = F.interpolate(im1_1, scale_factor=0.5, mode='bilinear', align_corners=self.cfg.align_corners)
        im2_1 = F.interpolate(im2_0, scale_factor=0.5, mode='bilinear', align_corners=self.cfg.align_corners)
        im2_2 = F.interpolate(im2_1, scale_factor=0.5, mode='bilinear', align_corners=self.cfg.align_corners)

        # Forward -----------
        im1_gx, im1_gy = image_grads(im1_2.detach())
        weights1_x = torch.exp(-torch.mean(torch.abs(self.cfg.edge_constant * im1_gx), 1, keepdim=True))
        weights1_y = torch.exp(-torch.mean(torch.abs(self.cfg.edge_constant * im1_gy), 1, keepdim=True))

        flow12_gx, flow12_gy = image_grads(flow12_2)
        loss_smooth = self.cfg.w_smooth * (torch.mean(weights1_x * robust_l1(flow12_gx))
                                           + torch.mean(weights1_y * robust_l1(flow12_gy))) / 2.
        if self.cfg.with_bk:
            # Backward -----------
            im2_gx, im2_gy = image_grads(im2_2.detach())
            weights2_x = torch.exp(-torch.mean(torch.abs(self.cfg.edge_constant * im2_gx), 1, keepdim=True))
            weights2_y = torch.exp(-torch.mean(torch.abs(self.cfg.edge_constant * im2_gy), 1, keepdim=True))

            flow21_gx, flow21_gy = image_grads(flow21_2)
            loss_smooth += self.cfg.w_smooth * (torch.mean(weights2_x * robust_l1(flow21_gx))
                                               + torch.mean(weights2_y * robust_l1(flow21_gy))) / 2.

        # Calculate total loss #
        ########################
        total_loss = loss_warp + loss_smooth - loss_entropy

        return total_loss, loss_warp, loss_smooth, loss_entropy, inv_l1norm
