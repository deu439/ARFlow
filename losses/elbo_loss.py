import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss_blocks import SSIM, smooth_grad_1st, smooth_grad_2nd, TernaryLoss
from utils.warp_utils import flow_warp
from utils.warp_utils import get_occu_mask_bidirection, get_occu_mask_backward


class ElboLoss(nn.modules.Module):
    def __init__(self, cfg):
        super(ElboLoss, self).__init__()
        self.cfg = cfg
        self.Normal = torch.distributions.Normal(0, 1)
        self.Normal.loc = self.Normal.loc.cuda() # hack to get sampling on the GPU
        self.Normal.scale = self.Normal.scale.cuda()

    def reparam(self, mean, log_var, nsamples=1):
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

    def loss_photomatric(self, im1_scaled, im1_recons, occu_mask1):
        loss = []

        if self.cfg.w_l1 > 0:
            loss += [self.cfg.w_l1 * (im1_scaled - im1_recons).abs() * occu_mask1]

        if self.cfg.w_ssim > 0:
            loss += [self.cfg.w_ssim * SSIM(im1_recons * occu_mask1,
                                            im1_scaled * occu_mask1)]

        if self.cfg.w_ternary > 0:
            loss += [self.cfg.w_ternary * TernaryLoss(im1_recons * occu_mask1,
                                                      im1_scaled * occu_mask1)]

        return sum([l.mean() for l in loss]) / occu_mask1.mean()

    def loss_smooth(self, flow, im1_scaled):
        if 'smooth_2nd' in self.cfg and self.cfg.smooth_2nd:
            func_smooth = smooth_grad_2nd
        else:
            func_smooth = smooth_grad_1st
        loss = []
        loss += [func_smooth(flow, im1_scaled, self.cfg.alpha)]
        return sum([l.mean() for l in loss])

    def forward(self, output, target):
        """

        :param output: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :param target: image pairs Nx6xHxW
        :return:
        """

        pyramid_flows = output
        im1_origin = target[:, :3]
        im2_origin = target[:, 3:]

        pyramid_smooth_losses = []
        pyramid_warp_losses = []
        pyramid_entropy = []
        self.pyramid_occu_mask1 = []
        self.pyramid_occu_mask2 = []

        s = 1.
        for i, flow in enumerate(pyramid_flows):
            if self.cfg.w_scales[i] == 0:
                pyramid_warp_losses.append(0)
                pyramid_smooth_losses.append(0)
                continue

            b, _, h, w = flow.size()

            # resize images to match the size of layer
            im1_scaled = F.interpolate(im1_origin, (h, w), mode='area')
            im2_scaled = F.interpolate(im2_origin, (h, w), mode='area')

            # sample flows
            # flow[:, 0:2] .. forward flow
            # flow[:, 2:4] .. forward flow log variance
            # flow[:, 4:6] .. backward flow
            # flow[:, 6:8] .. backward flow log variance
            flow_sample_fw = self.reparam(flow[:, 0:2], flow[:, 2:4])
            flow_sample_bw = self.reparam(flow[:, 4:6], flow[:, 6:8])

            im1_recons = flow_warp(im2_scaled, flow_sample_fw, pad=self.cfg.warp_pad)
            im2_recons = flow_warp(im1_scaled, flow_sample_bw, pad=self.cfg.warp_pad)

            if i == 0:
                if self.cfg.occ_from_back:
                    occu_mask1 = 1 - get_occu_mask_backward(flow_sample_bw, th=0.2)
                    occu_mask2 = 1 - get_occu_mask_backward(flow_sample_fw, th=0.2)
                else:
                    occu_mask1 = 1 - get_occu_mask_bidirection(flow_sample_fw, flow_sample_bw)
                    occu_mask2 = 1 - get_occu_mask_bidirection(flow_sample_bw, flow_sample_fw)
            else:
                occu_mask1 = F.interpolate(self.pyramid_occu_mask1[0], (h, w), mode='nearest')
                occu_mask2 = F.interpolate(self.pyramid_occu_mask2[0], (h, w), mode='nearest')

            self.pyramid_occu_mask1.append(occu_mask1)
            self.pyramid_occu_mask2.append(occu_mask2)

            loss_warp = self.loss_photomatric(im1_scaled, im1_recons, occu_mask1)

            if i == 0:
                s = min(h, w)

            loss_smooth = self.loss_smooth(flow_sample_fw / s, im1_scaled)

            entropy = torch.sum(flow[:, 2:4], dim=1).mean() / 2.0

            if self.cfg.with_bk:
                loss_warp += self.loss_photomatric(im2_scaled, im2_recons,
                                                   occu_mask2)
                loss_smooth += self.loss_smooth(flow_sample_bw / s, im2_scaled)

                entropy += torch.sum(flow[:, 6:8], dim=1).mean() / 2.0

                loss_warp /= 2.0
                loss_smooth /= 2.0
                entropy /= 2.0      # why not

            pyramid_warp_losses.append(loss_warp)
            pyramid_smooth_losses.append(loss_smooth)
            pyramid_entropy.append(entropy)

        pyramid_warp_losses = [l * w for l, w in
                               zip(pyramid_warp_losses, self.cfg.w_scales)]
        pyramid_smooth_losses = [l * w for l, w in
                                 zip(pyramid_smooth_losses, self.cfg.w_sm_scales)]
        pyramid_entropy = [l * w for l, w in
                                 zip(pyramid_entropy, self.cfg.w_en_scales)]

        warp_loss = sum(pyramid_warp_losses)
        smooth_loss = self.cfg.w_smooth * sum(pyramid_smooth_losses)
        entropy = self.cfg.w_entropy * sum(pyramid_entropy)
        total_loss = warp_loss + smooth_loss - entropy      # We seek maximum entropy solution

        return total_loss, warp_loss, smooth_loss, entropy, pyramid_flows[0].abs().mean()
