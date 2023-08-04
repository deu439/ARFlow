import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect


def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


def get_corresponding_map(data):
    """

    :param data: unnormalized coordinates Bx2xHxW
    :return: Bx1xHxW
    """
    B, _, H, W = data.size()

    # x = data[:, 0, :, :].view(B, -1).clamp(0, W - 1)  # BxN (N=H*W)
    # y = data[:, 1, :, :].view(B, -1).clamp(0, H - 1)

    x = data[:, 0, :, :].view(B, -1)  # BxN (N=H*W)
    y = data[:, 1, :, :].view(B, -1)

    # invalid = (x < 0) | (x > W - 1) | (y < 0) | (y > H - 1)   # BxN
    # invalid = invalid.repeat([1, 4])

    x1 = torch.floor(x)
    x_floor = x1.clamp(0, W - 1)
    y1 = torch.floor(y)
    y_floor = y1.clamp(0, H - 1)
    x0 = x1 + 1
    x_ceil = x0.clamp(0, W - 1)
    y0 = y1 + 1
    y_ceil = y0.clamp(0, H - 1)

    x_ceil_out = x0 != x_ceil
    y_ceil_out = y0 != y_ceil
    x_floor_out = x1 != x_floor
    y_floor_out = y1 != y_floor
    invalid = torch.cat([x_ceil_out | y_ceil_out,
                         x_ceil_out | y_floor_out,
                         x_floor_out | y_ceil_out,
                         x_floor_out | y_floor_out], dim=1)

    # encode coordinates, since the scatter function can only index along one axis
    corresponding_map = torch.zeros(B, H * W).type_as(data)
    indices = torch.cat([x_ceil + y_ceil * W,
                         x_ceil + y_floor * W,
                         x_floor + y_ceil * W,
                         x_floor + y_floor * W], 1).long()  # BxN   (N=4*H*W)
    values = torch.cat([(1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_floor)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_floor))],
                       1)
    # values = torch.ones_like(values)

    values[invalid] = 0

    corresponding_map.scatter_add_(1, indices, values)
    # decode coordinates
    corresponding_map = corresponding_map.view(B, H, W)

    return corresponding_map.unsqueeze(1)


def flow_warp(x, flow12, pad='border', mode='bilinear'):
    B, _, H, W = x.size()

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

    v_grid = norm_grid(base_grid + flow12)  # BHW2
    if 'align_corners' in inspect.getfullargspec(torch.nn.functional.grid_sample).args:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)
    else:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    return im1_recons


def get_occu_mask_bidirection(flow12, flow21, scale=0.01, bias=0.5):
    flow21_warped = flow_warp(flow21, flow12, pad='zeros')
    flow12_diff = flow12 + flow21_warped
    mag = (flow12 * flow12).sum(1, keepdim=True) + \
          (flow21_warped * flow21_warped).sum(1, keepdim=True)
    occ_thresh = scale * mag + bias
    occ = (flow12_diff * flow12_diff).sum(1, keepdim=True) > occ_thresh
    return occ.float()


def get_occu_mask_backward(flow21, th=0.2):
    """
    The function returns a mask which is 1 (or close to 1) at occluded pixels!
    """
    B, _, H, W = flow21.size()
    base_grid = mesh_grid(B, H, W).type_as(flow21)  # B2HW

    corr_map = get_corresponding_map(base_grid + flow21)  # BHW
    if th > 0:
        occu_mask = corr_map.clamp(min=0., max=1.) < th
    else:
        occu_mask = 1. - corr_map.clamp(min=0., max=1.).detach()

    return occu_mask.float()


def border_mask(flow):
    """
    Generates a mask that is True for pixels whose correspondence is inside the image borders.
    flow: optical flow tensor (batch, 2, height, width)
    returns: mask (batch, 1, height, width)
    """
    b, _, h, w = flow.size()
    x = torch.arange(w).type_as(flow)
    y = torch.arange(h).type_as(flow)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    Xp = X.view(1, h, w).repeat(b, 1, 1) + flow[:, 0, :, :]
    Yp = Y.view(1, h, w).repeat(b, 1, 1) + flow[:, 1, :, :]
    mask_x = (Xp > 0.0) & (Xp < w-1.0)
    mask_y = (Yp > 0.0) & (Yp < h-1.0)

    return (mask_x & mask_y).view(b, 1, h, w).float()


# Credit: https://github.com/google-research/google-research/blob/master/uflow/uflow_utils.py#L113
def flow_to_warp(flow):
    # Convert to uflow axes and channels order
    flow = flow.permute(0,2,3,1)
    flow = flow.flip(-1)

    # Construct a grid of the image coordinates.
    _, height, width, _ = flow.size()
    i_grid, j_grid = torch.meshgrid(
        torch.linspace(0.0, height - 1.0, int(height)),
        torch.linspace(0.0, width - 1.0, int(width)),
        indexing='ij')
    grid = torch.stack([i_grid, j_grid], axis=2)

    # Add the flow field to the image grid.
    grid = grid.type_as(flow)
    warp = grid + flow
    return warp


# Credit: https://github.com/google-research/google-research/blob/master/uflow/uflow_utils.py#L113
def compute_range_map(flow):
    """Count how often each coordinate is sampled.

    Counts are assigned to the integer coordinates around the sampled coordinates
    using weights from bilinear interpolation.

    Args:
      flow: A float tensor of shape (batch size x height x width x 2) that
        represents a dense flow field.

    Returns:
      A float tensor of shape [batch_size, height, width, 1] that denotes how
      often each pixel is sampled.
    """

    # Get input shape.
    assert flow.dim() == 4
    batch_size, _, input_height, input_width = flow.size()

    flow_height = input_height
    flow_width = input_width

    # Move the coordinate frame appropriately
    output_height = input_height
    output_width = input_width
    coords = flow_to_warp(flow)

    # Split coordinates into an integer part and a float offset for interpolation.
    coords_floor = torch.floor(coords)
    coords_offset = coords - coords_floor
    coords_floor = coords_floor.long()

    # Define a batch offset for flattened indexes into all pixels.
    batch_range = torch.reshape(torch.arange(batch_size), (batch_size, 1, 1)).type_as(coords_floor)
    idx_batch_offset = torch.tile(batch_range, [1, flow_height, flow_width]) * output_height * output_width

    # Flatten everything.
    coords_floor_flattened = torch.reshape(coords_floor, [-1, 2])
    coords_offset_flattened = torch.reshape(coords_offset, [-1, 2])
    idx_batch_offset_flattened = torch.reshape(idx_batch_offset, [-1])

    # Initialize results.
    idxs_list = []
    weights_list = []

    # Loop over differences di and dj to the four neighboring pixels.
    for di in range(2):
        for dj in range(2):

            # Compute the neighboring pixel coordinates.
            idxs_i = coords_floor_flattened[:, 0] + di
            idxs_j = coords_floor_flattened[:, 1] + dj
            # Compute the flat index into all pixels.
            idxs = idx_batch_offset_flattened + idxs_i * output_width + idxs_j

            # Only count valid pixels.
            mask = (idxs_i >= 0) & (idxs_i < output_height) & (idxs_j >= 0) & (idxs_j < output_width)
            mask = torch.stack(torch.where(mask)).T
            mask = torch.reshape(mask, [-1])
            valid_idxs = idxs[mask]     # FIXME
            valid_offsets = coords_offset_flattened[mask]   # FIXME

            # Compute weights according to bilinear interpolation.
            weights_i = (1. - di) - (-1)**di * valid_offsets[:, 0]
            weights_j = (1. - dj) - (-1)**dj * valid_offsets[:, 1]
            weights = weights_i * weights_j

            # Append indices and weights to the corresponding list.
            idxs_list.append(valid_idxs)
            weights_list.append(weights)

    # Concatenate everything.
    idxs = torch.concat(idxs_list, axis=0)
    weights = torch.concat(weights_list, axis=0)

    # Sum up weights for each pixel and reshape the result.
    num_segments = batch_size * output_height * output_width
    counts = torch.zeros(num_segments, dtype=weights.dtype, device=weights.device)
    counts.scatter_add_(0, idxs, weights)   # FIXME
    count_image = torch.reshape(counts, [batch_size, 1, output_height, output_width])

    return count_image
