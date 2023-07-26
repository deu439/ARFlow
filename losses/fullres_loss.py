import torch.nn as nn
import torch.nn.functional as F
from .loss_blocks import SSIM, smooth_grad_1st, smooth_grad_2nd, TernaryLoss
from utils.warp_utils import flow_warp
from utils.warp_utils import get_occu_mask_bidirection, get_occu_mask_backward, border_mask
from utils.flow_utils import resize_flow


class FullResLoss(nn.modules.Module):
    def __init__(self, cfg):
        super(FullResLoss, self).__init__()
        self.cfg = cfg

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

        flow12_0 = output[0][:, 0:2]
        flow21_0 = output[0][:, 2:4]
        flow12_1 = output[1][:, 0:2]
        flow21_1 = output[1][:, 2:4]
        im1 = target[:, :3]
        im2 = target[:, 3:]

        # warp the images #
        ###################
        im1_recons = flow_warp(im2, flow12_0, pad=self.cfg.warp_pad)
        im2_recons = flow_warp(im1, flow21_0, pad=self.cfg.warp_pad)

        # Calculate border and occlusion masks #
        ########################################
        border_mask1 = border_mask(flow12_0)
        border_mask2 = border_mask(flow21_0)

        if self.cfg.occ_from_back:
            occu_mask1 = 1. - get_occu_mask_backward(flow12_0, th=self.cfg.wang_thr)
            occu_mask2 = 1. - get_occu_mask_backward(flow21_0, th=self.cfg.wang_thr)
        else:
            occu_mask1 = 1. - get_occu_mask_bidirection(flow12_0, flow21_0)
            occu_mask2 = 1. - get_occu_mask_bidirection(flow21_0, flow12_0)

        # Calculate photometric loss on the full-resolution images #
        ############################################################
        loss_warp = self.loss_photomatric(im1, im1_recons, occu_mask1 * border_mask1)
        if self.cfg.with_bk:
            loss_warp += self.loss_photomatric(im2, im2_recons, occu_mask2 * border_mask2)

        # Calculate smoothness loss at the scale of the last layer #
        ############################################################
        _, _, h, w = flow12_1.size()
        im1s = F.interpolate(im1, (h, w), mode='area')
        im2s = F.interpolate(im2, (h, w), mode='area')
        loss_smooth = self.loss_smooth(flow12_1, im1s)
        if self.cfg.with_bk:
            loss_smooth += self.loss_smooth(flow21_1, im2s)

        # Calculate total loss #
        ########################
        total_loss = loss_warp + self.cfg.w_smooth * loss_smooth

        return total_loss, loss_warp, loss_smooth, output[0].abs().mean()
