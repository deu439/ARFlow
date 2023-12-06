import torch
import torch.nn as nn

from utils.triag_solve import BackwardSubst, inverse_l1norm, matrix_vector_product, matrix_vector_product_T
from utils.flow_utils import resize_flow


class MseLoss(nn.modules.Module):
    def __init__(self, cfg):
        super(MseLoss, self).__init__()
        self.cfg = cfg

        self.Normal = torch.distributions.Normal(0, 1)
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

    def forward(self, output, target):
        """

        :param output: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :return:
        """

        # Get level 2 outputs - before upsampling
        if self.cfg.diag:
            mean12_2 = output[2][:, 0:2]
            log_diag12_2 = output[2][:, 2:4]
        else:
            mean12_2 = output[2][:, 0:2]
            log_diag12_2 = output[2][:, 2:4]
            b, c, h, w = mean12_2.size()
            #left12_2 = torch.zeros([b, c, h, w-1], device=mean12_2.device)
            #over12_2 = torch.zeros([b, c, h-1, w], device=mean12_2.device)
            left12_2 = output[2][:, 4:6, :, :-1]
            over12_2 = output[2][:, 6:8, :-1, :]

            # Ensure that the matrices are diagonally dominant
            diag12_2 = torch.exp(log_diag12_2)
            if self.cfg.diag_dominant:
                diag12_2 = diag12_2 \
                           + torch.nn.functional.pad(torch.abs(left12_2), (1, 0)) \
                           + torch.nn.functional.pad(torch.abs(over12_2), (0, 0, 1, 0))

        # Calculate l1 norm of the precision matrix inverse #
        #####################################################
        inv_l1norm = 0.0
        #if not self.cfg.diag:
        #    K, L = mean12_2.shape[0:2]
        #    for k in range(K):
        #        for l in range(L):
        #            out12 = inverse_l1norm(diag12_2[k, l], left12_2[k, l], over12_2[k, l]).item()
        #            inv_l1norm = max(inv_l1norm, out12)

        # Reparametrization trick #
        ###########################
        if self.cfg.diag and not self.cfg.inv_cov:
            flow12_2 = self.reparam_diag(mean12_2, log_diag12_2)
        elif self.cfg.diag and self.cfg.inv_cov:
            flow12_2 = self.reparam_diag_inv(mean12_2, log_diag12_2)
        elif not self.cfg.diag and not self.cfg.inv_cov:
            flow12_2 = self.reparam_triag(mean12_2, diag12_2, left12_2, over12_2)
        elif not self.cfg.diag and self.cfg.inv_cov:
            flow12_2 = self.reparam_triag_inv(mean12_2, diag12_2, left12_2, over12_2)

        # Calculate entropy loss #
        ##########################
        if self.cfg.diag and not self.cfg.inv_cov:
            loss_entropy = self.cfg.w_entropy * torch.sum(log_diag12_2, dim=1).mean()
        elif self.cfg.diag and self.cfg.inv_cov:
            loss_entropy = -self.cfg.w_entropy * torch.sum(log_diag12_2, dim=1).mean()
        elif not self.cfg.diag and not self.cfg.inv_cov:
            loss_entropy = self.cfg.w_entropy * torch.sum(log_diag12_2, dim=1).mean()
        elif not self.cfg.diag and self.cfg.inv_cov:
            if self.cfg.approx_entropy:
                tmp12 = matrix_vector_product_T(diag12_2.detach(), left12_2.detach(), over12_2.detach(),
                                                flow12_2 - mean12_2.detach())
                loss_entropy = self.cfg.w_entropy * torch.sum(tmp12*tmp12/2, dim=1).mean()
            else:
                loss_entropy = -self.cfg.w_entropy * torch.sum(log_diag12_2, dim=1).mean()

        # Resize the ground-truth flow & calculate loss
        _, _, height, width = flow12_2.size()
        gt_flow12_2 = resize_flow(target, new_shape=(height, width), align_corners=self.cfg.align_corners)
        loss_mse = self.cfg.w_mse * torch.mean(torch.square(flow12_2 - gt_flow12_2))

        # Calculate total loss #
        ########################
        total_loss = loss_mse - loss_entropy

        return total_loss, loss_mse, loss_entropy, inv_l1norm
