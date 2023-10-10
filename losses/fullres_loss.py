import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss_blocks import SSIM, smooth_grad_1st, smooth_grad_2nd, TernaryLoss, penalty_ddflow
from utils.warp_utils import flow_warp, compute_range_map
from utils.warp_utils import get_occu_mask_bidirection, get_occu_mask_backward, border_mask


class FullResLoss(nn.modules.Module):
    def __init__(self, cfg):
        super(FullResLoss, self).__init__()
        self.cfg = cfg

    def loss_photometric(self, im1_scaled, im1_recons, occu_mask1):
        loss = 0

        if self.cfg.w_l1 > 0:
            l1_loss = self.cfg.w_l1 * (im1_scaled - im1_recons).abs() * occu_mask1
            loss += torch.sum(l1_loss) / (torch.sum(occu_mask1) + 1e-6)

        if self.cfg.w_ssim > 0:
            ssim_loss = self.cfg.w_ssim * SSIM(im1_recons, im1_scaled) * occu_mask1
            loss += torch.sum(ssim_loss) / (torch.sum(occu_mask1) + 1e-6)

        if self.cfg.w_ternary > 0:
            dist, valid_mask = TernaryLoss(im1_scaled, im1_recons, max_distance=self.cfg.ternary_distance, sum_dist=True)
            mask = torch.detach(valid_mask * occu_mask1)
            ternary_loss = self.cfg.w_ternary * penalty_ddflow(dist) * mask
            loss += torch.sum(ternary_loss) / (torch.sum(mask) + 1e-6)

        return loss

    def loss_smooth(self, flow, im1_scaled):
        if 'smooth_2nd' in self.cfg and self.cfg.smooth_2nd:
            func_smooth = smooth_grad_2nd
        else:
            func_smooth = smooth_grad_1st
        loss = func_smooth(flow, im1_scaled, self.cfg.alpha, penalty="uflow") * 2.0
        return loss

    def forward(self, output, target):
        """

        :param output: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :param target: image pairs Nx6xHxW
        :return:
        """

        flow12_0 = output[0][:, 0:2]
        flow21_0 = output[0][:, 2:4]
        flow12_2 = output[2][:, 0:2]
        flow21_2 = output[2][:, 2:4]
        im1 = target[:, :3]
        im2 = target[:, 3:]

        # Warp the images #
        ###################
        im1_recons = flow_warp(im2.detach(), flow12_0, pad=self.cfg.warp_pad, align_corners=self.cfg.align_corners)
        if self.cfg.with_bk:
            im2_recons = flow_warp(im1.detach(), flow21_0, pad=self.cfg.warp_pad, align_corners=self.cfg.align_corners)

        # Calculate border and occlusion masks #
        ########################################
        border_mask1 = border_mask(flow12_0)
        if self.cfg.with_bk:
            border_mask2 = border_mask(flow21_0)

        if self.cfg.occ_type == 'wang':
            occu_mask1 = 1. - get_occu_mask_backward(flow21_0, th=self.cfg.wang_thr)
            occu_mask2 = 1. - get_occu_mask_backward(flow12_0, th=self.cfg.wang_thr)
        elif self.cfg.occ_type == 'wang1':
            occu_mask1 = torch.clamp(compute_range_map(flow21_0), min=0., max=1.)
            occu_mask2 = torch.clamp(compute_range_map(flow12_0), min=0., max=1.)
        elif self.cfg.occ_type == 'brox':
            occu_mask1 = 1. - get_occu_mask_bidirection(flow12_0, flow21_0)
            occu_mask2 = 1. - get_occu_mask_bidirection(flow21_0, flow12_0)
        elif self.cfg.occ_type == 'none':
            occu_mask1 = torch.ones_like(flow12_0)
            occu_mask2 = torch.ones_like(flow21_0)
        else:
            raise NotImplementedError(self.cfg.occ_type)

        # Calculate photometric loss on the full-resolution images #
        ############################################################
        loss_warp = self.loss_photometric(im1, im1_recons, occu_mask1 * border_mask1)
        if self.cfg.with_bk:
            loss_warp += self.loss_photometric(im2, im2_recons, occu_mask2 * border_mask2)

        # Calculate smoothness loss at the scale of the last layer #
        ############################################################
        _, _, h, w = flow12_2.size()
        im1_2 = F.interpolate(im1, (h, w), mode='bilinear', align_corners=self.cfg.align_corners)
        im2_2 = F.interpolate(im2, (h, w), mode='bilinear', align_corners=self.cfg.align_corners)
        # Resample twice as in uflow
        #im1_1 = F.interpolate(im1, scale_factor=0.5, mode='area')
        #im1_2 = F.interpolate(im1_1, scale_factor=0.5, mode='area')
        #im2_1 = F.interpolate(im2, scale_factor=0.5, mode='area')
        #im2_2 = F.interpolate(im2_1, scale_factor=0.5, mode='area')
        loss_smooth = self.loss_smooth(flow12_2, im1_2.detach())
        if self.cfg.with_bk:
            loss_smooth += self.loss_smooth(flow21_2, im2_2.detach())

        # Calculate total loss #
        ########################
        total_loss = loss_warp + self.cfg.w_smooth * loss_smooth

        return total_loss, loss_warp, loss_smooth, output[0].abs().mean()
