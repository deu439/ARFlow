import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.uflow_utils import flow_to_warp, resample, mask_invalid, census_loss, image_grads, robust_l1, upsample, downsample
from utils.warp_utils import compute_range_map


class UFlowLoss(nn.modules.Module):
    def __init__(self, cfg):
        super(UFlowLoss, self).__init__()
        self.cfg = cfg

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
        im1_0 = target[:, :3]
        im2_0 = target[:, 3:]

        # Warp the images #
        ###################
        warp12_0 = flow_to_warp(flow12_0)
        im1_recons = resample(im2_0.detach(), warp12_0)
        if self.cfg.with_bk:
            warp21_0 = flow_to_warp(flow21_0)
            im2_recons = resample(im1_0.detach(), warp21_0)

        # Calculate border and occlusion masks #
        ########################################
        valid_mask1 = mask_invalid(warp12_0)
        # Calculate occlusion masks at level 2 and then upsample. Computing occlusion mask at bilinear upsampled
        # images would produce artifacts!
        occu_mask1 = torch.clamp(compute_range_map(flow21_2), min=0., max=1.)
        occu_mask1 = upsample(occu_mask1, is_flow=False, scale_factor=4.0)
        mask1 = torch.detach(occu_mask1 * valid_mask1)
        if self.cfg.with_bk:
            valid_mask2 = mask_invalid(warp21_0)
            occu_mask2 = torch.clamp(compute_range_map(flow12_2), min=0., max=1.)
            occu_mask2 = upsample(occu_mask2, is_flow=False, scale_factor=4.0)
            mask2 = torch.detach(occu_mask2 * valid_mask2)

        # Calculate photometric loss on level 0 #
        ############################################################
        loss_warp = self.cfg.w_census * census_loss(im1_0, im1_recons, mask1)
        if self.cfg.with_bk:
            loss_warp += self.cfg.w_census * census_loss(im2_0, im2_recons, mask2)

        # Calculate smoothness loss on level 2 #
        ############################################################
        _, _, height, width = im1_0.size()
        im1_2 = downsample(im1_0, is_flow=False, scale_factor=4.0)
        im2_2 = downsample(im2_0, is_flow=False, scale_factor=4.0)

        if self.cfg.smooth_order == 1:
            # Forward -----------
            im1_gx, im1_gy = image_grads(im1_2.detach())
            weights1_x = torch.exp(-torch.mean(torch.abs(self.cfg.edge_constant * im1_gx), 1, keepdim=True))
            weights1_y = torch.exp(-torch.mean(torch.abs(self.cfg.edge_constant * im1_gy), 1, keepdim=True))

            flow12_gx, flow12_gy = image_grads(flow12_2)
            loss_smooth = self.cfg.w_smooth * (torch.mean(weights1_x * robust_l1(flow12_gx**2))
                                               + torch.mean(weights1_y * robust_l1(flow12_gy**2))) / 2.
            if self.cfg.with_bk:
                # Backward -----------
                im2_gx, im2_gy = image_grads(im2_2.detach())
                weights2_x = torch.exp(-torch.mean(torch.abs(self.cfg.edge_constant * im2_gx), 1, keepdim=True))
                weights2_y = torch.exp(-torch.mean(torch.abs(self.cfg.edge_constant * im2_gy), 1, keepdim=True))

                flow21_gx, flow21_gy = image_grads(flow21_2)
                loss_smooth += self.cfg.w_smooth * (torch.mean(weights2_x * robust_l1(flow21_gx**2))
                                                   + torch.mean(weights2_y * robust_l1(flow21_gy**2))) / 2.

        if self.cfg.smooth_order == 2:
            # Forward -----------
            im1_gx, im1_gy = image_grads(im1_2.detach(), stride=2)
            weights1_x = torch.exp(-torch.mean(torch.abs(self.cfg.edge_constant * im1_gx), 1, keepdim=True))
            weights1_y = torch.exp(-torch.mean(torch.abs(self.cfg.edge_constant * im1_gy), 1, keepdim=True))

            flow12_gx, flow12_gy = image_grads(flow12_2)
            flow12_gxx, _ = image_grads(flow12_gx)
            _, flow12_gyy = image_grads(flow12_gy)
            loss_smooth = self.cfg.w_smooth * (torch.mean(weights1_x * robust_l1(flow12_gxx**2))
                                               + torch.mean(weights1_y * robust_l1(flow12_gyy**2))) / 2.
            if self.cfg.with_bk:
                # Backward -----------
                im2_gx, im2_gy = image_grads(im2_2.detach(), stride=2)
                weights2_x = torch.exp(-torch.mean(torch.abs(self.cfg.edge_constant * im2_gx), 1, keepdim=True))
                weights2_y = torch.exp(-torch.mean(torch.abs(self.cfg.edge_constant * im2_gy), 1, keepdim=True))

                flow21_gx, flow21_gy = image_grads(flow21_2)
                flow21_gxx, _ = image_grads(flow21_gx)
                _, flow21_gyy = image_grads(flow21_gy)
                loss_smooth += self.cfg.w_smooth * (torch.mean(weights2_x * robust_l1(flow21_gxx**2))
                                                    + torch.mean(weights2_y * robust_l1(flow21_gyy**2))) / 2.


        # Calculate total loss #
        ########################
        total_loss = loss_warp + loss_smooth

        return total_loss, loss_warp, loss_smooth, output[0].abs().mean(), mask1
