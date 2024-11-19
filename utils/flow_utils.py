import torch
import cv2
import numpy as np
from matplotlib.colors import hsv_to_rgb
import scipy.special as ss
import math

def load_flow(path):
    if path.endswith('.png'):
        # for KITTI which uses 16bit PNG images
        # see 'https://github.com/ClementPinard/FlowNetPytorch/blob/master/datasets/KITTI.py'
        # The -1 is here to specify not to change the image depth (16bit), and is compatible
        # with both OpenCV2 and OpenCV3
        flo_file = cv2.imread(path, -1)
        flo_img = flo_file[:, :, 2:0:-1].astype(np.float32)
        invalid = (flo_file[:, :, 0] == 0)  # mask
        flo_img = flo_img - 32768
        flo_img = flo_img / 64
        flo_img[np.abs(flo_img) < 1e-10] = 1e-10
        flo_img[invalid, :] = 0
        return flo_img, np.expand_dims(flo_file[:, :, 0], 2)
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

        flo_pred = cv2.resize(pred_flow, (W, H), interpolation=cv2.INTER_LINEAR)

        epe_map = np.sqrt(
            np.sum(np.square(flo_pred[:, :, :2] - gt_flow[:, :, :2]), axis=2))
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


def sp_plot(error, entropy, n=25, alpha=100.0, eps=1e-1):
    def sp_mask(thr, entropy):
        mask = ss.expit(alpha * (thr[:, None, None] - entropy[None, :, :]))
        frac = np.mean(1.0 - mask, axis=(1, 2))
        return mask, frac

    # Find the primary interval for soft thresholding
    greatest = np.max(entropy) + eps    # Avoid zero-sized interval
    least = np.min(entropy) - eps
    _, frac = sp_mask(np.array([least]), entropy)
    while abs(frac.item() - 1.0) > eps:
        least -= 1e-3*(greatest - least)
        _, frac = sp_mask(np.array([least]), entropy)

    _, frac = sp_mask(np.array([greatest]), entropy)
    while abs(frac.item() - 0.0) > eps:
        greatest += 1e-3*(greatest - least)
        _, frac = sp_mask(np.array([greatest]), entropy)

    # Approximate uniform grid
    grid_entr = np.linspace(greatest, least, n)
    grid_frac = np.linspace(0, 1, n)
    mask, frac = sp_mask(grid_entr, entropy)
    for i in range(10):
        #print("res: ", np.max(np.abs(frac - grid_frac)))
        if np.max(np.abs(frac - grid_frac)) <= eps:
            break
        grid_entr = np.interp(grid_frac, frac, grid_entr)
        mask, frac = sp_mask(grid_entr, entropy)

    # Check whether the grid is approximately uniform
    if np.max(np.abs(frac - grid_frac)) > eps:
        print("Warning! sp_plot did not converge!")
        #raise RuntimeError("sp_plot did not converge!")

    # Calculate the sparsification plot
    splot = np.sum(error[None, :, :] * mask, axis=(1,2)) / np.sum(mask, axis=(1,2))

    # Resample on uniform grid
    splot = np.interp(grid_frac, frac, splot)

    return splot


def evaluate_uncertainty(gt_flows, pred_flows, pred_entropies, sp_samples=25):
    sauc, oauc = 0, 0
    splots, oplots = [], []
    B = len(gt_flows)
    for gt_flow, pred_flow, pred_entropy, i in zip(gt_flows, pred_flows, pred_entropies, range(B)):
        H, W = gt_flow.shape[:2]

        # Resample flow
        h, w = pred_flow.shape[:2]
        pred_flow = np.copy(pred_flow)
        pred_flow[:, :, 0] = pred_flow[:, :, 0] / w * W
        pred_flow[:, :, 1] = pred_flow[:, :, 1] / h * H
        flo_pred = cv2.resize(pred_flow, (W, H), interpolation=cv2.INTER_LINEAR)

        # Resample entropy
        pred_entropy = np.copy(pred_entropy)
        pred_entropy[:, :, 0] = pred_entropy[:, :, 0] - 2*math.log(w) + 2*math.log(W)
        pred_entropy[:, :, 1] = pred_entropy[:, :, 1] - 2*math.log(h) + 2*math.log(H)
        pred_entropy = cv2.resize(pred_entropy, (W, H), interpolation=cv2.INTER_LINEAR)

        # Calculate sparsification plots
        epe_map = np.sqrt(np.sum(np.square(flo_pred[:, :, :2] - gt_flow[:, :, :2]), axis=2))
        entropy_map = np.sum(pred_entropy[:, :, :2], axis=2)
        splot = sp_plot(epe_map, entropy_map)
        oplot = sp_plot(epe_map, epe_map)     # Oracle

        # Collect the splots and oplots
        splots += [splot]
        oplots += [oplot]

        #import matplotlib.pyplot as plt
        #plt.plot(sfrac, splot, '+-')
        #plt.show()

        # Cummulate AUC
        frac = np.linspace(0, 1, sp_samples)
        sauc += np.trapz(splot / splot[0], x=frac)
        oauc += np.trapz(oplot / oplot[0], x=frac)

    return [sauc / B, (sauc - oauc) / B], splots, oplots

