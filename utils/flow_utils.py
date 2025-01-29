import torch
import cv2
import numpy as np
from matplotlib.colors import hsv_to_rgb
import scipy.special as ss
import math
from collections import defaultdict

def load_flow(path):
    if path.endswith('.png'):
        # for KITTI which uses 16bit PNG images
        # see 'https://github.com/ClementPinard/FlowNetPytorch/blob/master/datasets/KITTI.py'
        # The -1 is here to specify not to change the image depth (16bit), and is compatible
        # with both OpenCV2 and OpenCV3
        flo_file = cv2.imread(path, -1).astype(np.float32)
        flo_img = flo_file[:, :, 2:0:-1]
        mask = flo_file[:, :,[0]]  # mask
        flo_img = flo_img - 32768
        flo_img = flo_img / 64
        flo_img[np.abs(flo_img) < 1e-10] = 1e-10
        flo_img = flo_img * mask
        return np.concatenate([flo_img, mask], axis=-1)
    else:
        with open(path, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
            h = np.fromfile(f, np.int32, count=1)[0]
            w = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2 * w * h)
        # Reshape data into 3D array (columns, rows, bands)
        data2D = np.resize(data, (w, h, 2))
        return data2D


def write_flow(filename, uv, v=None):
    """ Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2
    TAG_CHAR = np.array([202021.25], np.float32)

    if v is None:
        assert (uv.ndim == 3)
        assert (uv.shape[2] == 2)
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert (u.shape == v.shape)
    height, width = u.shape
    f = open(filename, 'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width * nBands))
    tmp[:, np.arange(width) * 2] = u
    tmp[:, np.arange(width) * 2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()

def flow_to_image(flow, max_flow=256):
    if max_flow is not None:
        max_flow = max(max_flow, 1.)
    else:
        max_flow = np.max(flow)

    n = 8
    u, v = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(v, u)
    im_h = np.mod(angle / (2 * np.pi) + 1, 1)
    im_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
    im_v = np.clip(n - im_s, a_min=0, a_max=1)
    im = hsv_to_rgb(np.stack([im_h, im_s, im_v], 2))
    return (im * 255).astype(np.uint8)


def np_flow2rgb(flow_map, max_value=None):
    _, h, w = flow_map.shape
    # flow_map[:,(flow_map[0] == 0) & (flow_map[1] == 0)] = float('nan')
    # print np.any(np.isnan(flow_map))
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map / max_value
    else:
        normalized_flow_map = flow_map / (np.abs(flow_map).max())
    rgb_map[:, :, 0] += normalized_flow_map[0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[:, :, 2] += normalized_flow_map[1]
    return rgb_map.clip(0, 1)


def torch_flow2rgb(tensor):
    batch_size, channels, height, width = tensor.size()
    assert(channels == 2)
    array = np.empty((batch_size, height, width, 3))
    for i in range(batch_size):
        flow = np.array(tensor[i, :, :, :])
        array[i, :, :, :] = np_flow2rgb(flow)

    return torch.Tensor(np.transpose(array, (0, 3, 1, 2)))


def resize_flow(flow, new_shape, align_corners=False):
    _, _, h, w = flow.shape
    new_h, new_w = new_shape
    flow = torch.nn.functional.interpolate(flow, (new_h, new_w),
                                           mode='bilinear', align_corners=align_corners)
    scale_h, scale_w = h / float(new_h), w / float(new_w)
    flow[:, 0] /= scale_w
    flow[:, 1] /= scale_h
    return flow


def evaluate_flow(gt_flows, pred_flows, moving_masks=None):
    # credit "undepthflow/eval/evaluate_flow.py"
    def calculate_error_rate(epe_map, gt_flow, mask):
        bad_pixels = np.logical_and(
            epe_map * mask > 3,
            epe_map * mask / np.maximum(
                np.sqrt(np.sum(np.square(gt_flow), axis=2)), 1e-10) > 0.05)
        return bad_pixels.sum() / mask.sum() * 100.

    error, error_noc, error_occ, error_move, error_static, error_rate = \
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    error_move_rate, error_static_rate = 0.0, 0.0
    B = len(gt_flows)
    for gt_flow, pred_flow, i in zip(gt_flows, pred_flows, range(B)):
        H, W = gt_flow.shape[:2]

        h, w = pred_flow.shape[:2]
        pred_flow = np.copy(pred_flow)
        pred_flow[:, :, 0] = pred_flow[:, :, 0] / w * W
        pred_flow[:, :, 1] = pred_flow[:, :, 1] / h * H

        # Back to original size
        pred_flow = cv2.resize(pred_flow, (W, H), interpolation=cv2.INTER_LINEAR)

        epe_map = np.sqrt(
            np.sum(np.square(pred_flow[:, :, :2] - gt_flow[:, :, :2]), axis=2))
        if gt_flow.shape[-1] == 2:
            error += np.mean(epe_map)

        elif gt_flow.shape[-1] == 4:
            error += np.sum(epe_map * gt_flow[:, :, 2]) / np.sum(gt_flow[:, :, 2])
            noc_mask = gt_flow[:, :, -1]
            error_noc += np.sum(epe_map * noc_mask) / np.sum(noc_mask)

            error_occ += np.sum(epe_map * (gt_flow[:, :, 2] - noc_mask)) / max(
                np.sum(gt_flow[:, :, 2] - noc_mask), 1.0)

            error_rate += calculate_error_rate(epe_map, gt_flow[:, :, 0:2],
                                               gt_flow[:, :, 2])

            if moving_masks is not None:
                move_mask = moving_masks[i]

                error_move_rate += calculate_error_rate(
                    epe_map, gt_flow[:, :, 0:2], gt_flow[:, :, 2] * move_mask)
                error_static_rate += calculate_error_rate(
                    epe_map, gt_flow[:, :, 0:2],
                    gt_flow[:, :, 2] * (1.0 - move_mask))

                error_move += np.sum(epe_map * gt_flow[:, :, 2] *
                                     move_mask) / np.sum(gt_flow[:, :, 2] *
                                                         move_mask)
                error_static += np.sum(epe_map * gt_flow[:, :, 2] * (
                        1.0 - move_mask)) / np.sum(gt_flow[:, :, 2] *
                                                   (1.0 - move_mask))

    if gt_flows[0].shape[-1] == 4:
        res = [error / B, error_noc / B, error_occ / B, error_rate / B]
        if moving_masks is not None:
            res += [error_move / B, error_static / B]
        return res
    else:
        return [error / B]


def sp_plot(error, entropy, gt_mask, n=25, alpha=100.0, eps=1e-1):
    def sp_mask(thr, entropy, gt_mask):
        mask = ss.expit(alpha * (thr[:, None, None] - entropy[None, :, :]))
        frac = np.sum((1.0 - mask)*gt_mask[None], axis=(1, 2)) / np.sum(gt_mask)[None]
        return mask*gt_mask[None], frac

    # Find the primary interval for soft thresholding
    greatest = np.max(entropy) + eps    # Avoid zero-sized interval
    least = np.min(entropy) - eps
    _, frac = sp_mask(np.array([least]), entropy, gt_mask)
    while abs(frac.item() - 1.0) > eps:
        least -= 1e-3*(greatest - least)
        _, frac = sp_mask(np.array([least]), entropy, gt_mask)

    _, frac = sp_mask(np.array([greatest]), entropy, gt_mask)
    while abs(frac.item() - 0.0) > eps:
        greatest += 1e-3*(greatest - least)
        _, frac = sp_mask(np.array([greatest]), entropy, gt_mask)

    # Approximate uniform grid
    grid_entr = np.linspace(greatest, least, n)
    grid_frac = np.linspace(0, 1, n)
    mask, frac = sp_mask(grid_entr, entropy, gt_mask)
    for i in range(10):
        #print("res: ", np.max(np.abs(frac - grid_frac)))
        if np.max(np.abs(frac - grid_frac)) <= eps:
            break
        grid_entr = np.interp(grid_frac, frac, grid_entr)
        mask, frac = sp_mask(grid_entr, entropy, gt_mask)

    # Check whether the grid is approximately uniform
    if np.max(np.abs(frac - grid_frac)) > eps:
        print("Warning! sp_plot did not converge!")
        #raise RuntimeError("sp_plot did not converge!")

    # Calculate the sparsification plot
    splot = np.sum(error[None, :, :] * mask, axis=(1,2)) / np.sum(mask, axis=(1,2))

    # Resample on uniform grid
    splot = np.interp(grid_frac, frac, splot)

    return splot


class CalibrationCurve:
    def __init__(self, cc_max=2, cc_samples=25):
        self.cc_max = cc_max
        self.cc_samples = cc_samples
        self.errors = defaultdict(list)
        self.bins = np.linspace(0, self.cc_max, self.cc_samples)

    def __call__(self, gt_flows, pred_flows, pred_entropies):
        sigma = np.exp(pred_entropies - 1/2 * np.log(2*np.pi*np.exp(1)))
        bin_idx = np.digitize(sigma, self.bins)
        error = np.abs(gt_flows - pred_flows)
        
        for idx in range(self.cc_samples):
            self.errors[idx].extend(error[bin_idx == idx].reshape(-1))

    def calibration_curve(self):
        vals = list()  # middle value of bins
        means = list()  # should be close to vals
        sigmas = list()  # should be close to 0
                
        for idx in range(self.cc_samples):
            val = (idx+0.5)*self.cc_max/(self.cc_samples-1)
            mean = np.mean(self.errors[idx])
            var = np.var(self.errors[idx])
            sigma = np.sqrt(var)
            
            vals.append(val)
            means.append(mean)
            sigmas.append(sigma)
            
        return vals, means, sigmas
        


def evaluate_uncertainty(gt_flows, pred_flows, pred_entropies, sp_samples=25):
    auc, oracle_auc = 0, 0
    splots, oracle_splots = [], []
    batch_size = len(gt_flows)
    for gt_flow, pred_flow, pred_entropy, i in zip(gt_flows, pred_flows, pred_entropies, range(batch_size)):
        H, W = gt_flow.shape[:2]

        # Resample flow - back to original shape
        h, w = pred_flow.shape[:2]
        pred_flow = np.copy(pred_flow)
        pred_flow[:, :, 0] = pred_flow[:, :, 0] / w * W
        pred_flow[:, :, 1] = pred_flow[:, :, 1] / h * H
        pred_flow = cv2.resize(pred_flow, (W, H), interpolation=cv2.INTER_LINEAR)

        # Resample entropy - back to original shape
        pred_entropy = np.copy(pred_entropy)
        pred_entropy[:, :, 0] = pred_entropy[:, :, 0] - 2*math.log(w) + 2*math.log(W)
        pred_entropy[:, :, 1] = pred_entropy[:, :, 1] - 2*math.log(h) + 2*math.log(H)
        pred_entropy = cv2.resize(pred_entropy, (W, H), interpolation=cv2.INTER_LINEAR)

        # Calculate sparsification plots
        epe_map = np.sqrt(np.sum(np.square(pred_flow[:, :, :2] - gt_flow[:, :, :2]), axis=2))
        if gt_flow.shape[2] == 4:    # KITTY dataset includes masks in the third and fourth dimension
            mask = gt_flow[:, :, 2]
        else:
            mask = torch.ones_like(epe_map)
        entropy_map = np.sum(pred_entropy[:, :, :2], axis=2)
        splot = sp_plot(epe_map, entropy_map, mask)
        oracle_splot = sp_plot(epe_map, epe_map, mask)     # Oracle

        # Collect the sparsification plots and oracle sparsification plots
        splots += [splot]
        oracle_splots += [oracle_splot]

        # Cummulate AUC
        frac = np.linspace(0, 1, sp_samples)
        auc += np.trapz(splot / splot[0], x=frac)
        oracle_auc += np.trapz(oracle_splot / oracle_splot[0], x=frac)

    return [auc / batch_size, (auc - oracle_auc) / batch_size], splots, oracle_splots
