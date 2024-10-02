import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.uflow_resampler import resampler


def flow_to_warp(flow):
    """Compute the warp from the flow field.

    Args:
      [B, 2, H, W] flow: tf.tensor representing optical flow.

    Returns:
      [B, 2, H, W] The warp, i.e. the endpoints of the estimated flow.
    """

    # Construct a grid of the image coordinates.
    B, _, height, width = flow.shape
    j_grid, i_grid = torch.meshgrid(
        torch.linspace(0.0, height - 1.0, int(height)),
        torch.linspace(0.0, width - 1.0, int(width)))
    grid = torch.stack([i_grid, j_grid]).to(flow.device)

    # add batch dimension to match the shape of flow.
    grid = grid[None]
    grid = grid.repeat(B, 1, 1, 1)

    # Add the flow field to the image grid.
    if flow.dtype != grid.dtype:
        grid = grid.type(dtype=flow.dtype)
    warp = grid + flow
    return warp


def mask_invalid(coords):
    """Mask coordinates outside of the image.

    Valid = 1, invalid = 0.

    Args:
      coords: a 4D float tensor of image coordinates.

    Returns:
      The mask showing which coordinates are valid.
    """
    max_height = float(coords.shape[2] - 1)
    max_width = float(coords.shape[3] - 1)
    mask_y = (coords[:, 1, :, :] >= 0.0) & (coords[:, 1, :, :] <= max_height)
    mask_x = (coords[:, 0, :, :] >= 0.0) & (coords[:, 0, :, :] <= max_width)
    return (mask_x & mask_y).unsqueeze(1).float()


def resample2(source, coords):
    """Resample the source image at the passed coordinates.

    Args:
      source: tf.tensor, batch of images to be resampled.
      coords: [B, 2, H, W] tf.tensor, batch of coordinates in the image.

    Returns:
      The resampled image.

    Coordinates should be between 0 and size-1. Coordinates outside of this range
    are handled by interpolating with a background image filled with zeros in the
    same way that SAME size convolution works.
    """

    _, _, H, W = source.shape
    # normalize coordinates to [-1 .. 1] range
    coords = coords.clone()
    coords[:, 0, :, :] = 2.0 * coords[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    coords[:, 1, :, :] = 2.0 * coords[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    coords = coords.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(source, coords, align_corners=True, mode='bilinear', padding_mode='zeros')
    return output


def resample(source, coords):
    """Resample the source image at the passed coordinates.

    Args:
      source: [B, 3, H, W] batch of images to be resampled.
      coords: [B, 2, H, W] the warp, i.e. the endpoints of the estimated flow.

    Returns:
      The resampled images.

    Coordinates should be between 0 and size-1. Coordinates outside of this range
    are handled by interpolating with a background image filled with zeros in the
    same way that SAME size convolution works.
    """

    # Wrap this function because it uses a different order axes
    #orig_source_dtype = source.dtype
    #if source.dtype != tf.float32:
    #    source = tf.cast(source, tf.float32)
    #if coords.dtype != tf.float32:
    #    coords = tf.cast(coords, tf.float32)
    coords_rank = len(coords.shape)
    if coords_rank == 4:
        source_perm = source.permute(0,2,3,1)
        coords_perm = coords.permute(0,2,3,1)
        output = resampler(source_perm, coords_perm)
        return output.permute(0,3,1,2)
    else:
        raise NotImplementedError()


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
    coords = flow_to_warp(flow).permute(0,2,3,1).flip(-1)   # (B, C, U, V) -> (B, U, V, C)

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


def upsample(img, is_flow, scale_factor=2.0, align_corners=False):
    """Double resolution of an image or flow field.

    Args:
      img: [BCHW], image or flow field to be resized
      is_flow: bool, flag for scaling flow accordingly

    Returns:
      Resized and potentially scaled image or flow field.
    """

    img_resized = nn.functional.interpolate(img, scale_factor=scale_factor, mode='bilinear',
                                            align_corners=align_corners)

    if is_flow:
        # Scale flow values to be consistent with the new image size.
        img_resized *= scale_factor

    return img_resized


def downsample(img, is_flow, scale_factor=2.0, align_corners=False):
    """Double resolution of an image or flow field.

    Args:
      img: [BCHW], image or flow field to be resized
      is_flow: bool, flag for scaling flow accordingly

    Returns:
      Resized and potentially scaled image or flow field.
    """

    img_resized = nn.functional.interpolate(img, scale_factor=1/scale_factor, mode='bilinear',
                                            align_corners=align_corners)

    if is_flow:
        # Scale flow values to be consistent with the new image size.
        img_resized *= 1/scale_factor

    return img_resized


def image_grads(image_batch, stride=1):
    image_batch_x = image_batch[:, :, :, stride:] - image_batch[:, :, :, :-stride]
    image_batch_y = image_batch[:, :, stride:] - image_batch[:, :, :-stride]
    return image_batch_x, image_batch_y



def compute_loss(i1, warped2, flow, use_mag_loss=False):
    loss = torch.nn.functional.l1_loss(warped2, i1)
    if use_mag_loss:
        flow_mag = torch.sqrt(flow[:, 0] ** 2 + flow[:, 1] ** 2)
        mag_loss = flow_mag.mean()
        loss += mag_loss * 1e-4

    return loss


def rgb_to_grayscale(image):
    grayscale = image[:, 0, :, :] * 0.2989 + \
                image[:, 1, :, :] * 0.5870 + \
                image[:, 2, :, :] * 0.1140
    return grayscale.unsqueeze(1)


def zero_mask_border(mask, patch_size):
    """Used to ignore border effects from census_transform."""
    mask_padding = patch_size // 2
    mask = mask[:, :, mask_padding:-mask_padding, mask_padding:-mask_padding]
    return F.pad(mask, [mask_padding] * 4)


def census_transform(image, patch_size):
    """The census transform as described by DDFlow.

    See the paper at https://arxiv.org/abs/1902.09145

    Args:
      image: tensor of shape (b, c, h, w)
      patch_size: int
    Returns:
      image with census transform applied
    """
    intensities = rgb_to_grayscale(image) * 255

    out_channels = patch_size * patch_size
    kernel = torch.eye(out_channels).view((out_channels, 1, patch_size, patch_size))
    kernel = kernel.type_as(image)
    neighbors = F.conv2d(intensities, kernel, padding=patch_size//2)    # should be equivalent to padding SAME
    diff = neighbors - intensities
    # Coefficients adopted from DDFlow.
    diff_norm = diff / torch.sqrt(.81 + torch.square(diff))
    return diff_norm


def soft_hamming(a, b, thresh=.1):
    """A soft hamming distance between tensor a_bhwk and tensor b_bhwk.

    Args:
      a_bhwk: tf.Tensor of shape (batch, features, height, width)
      b_bhwk: tf.Tensor of shape (batch, features, height, width)
      thresh: float threshold

    Returns:
      a tensor with approx. 1 in (h, w) locations that are significantly
      more different than thresh and approx. 0 if significantly less
      different than thresh.
    """
    sq_dist = torch.square(a - b)
    soft_thresh_dist = sq_dist / (thresh + sq_dist)
    return torch.sum(soft_thresh_dist, 1, keepdim=True)  # uflow version


def census_loss(image_a, image_b, mask, patch_size=7):
    """Compares the similarity of the census transform of two images."""
    census_image_a = census_transform(image_a, patch_size)
    census_image_b = census_transform(image_b, patch_size)

    hamming = soft_hamming(census_image_a, census_image_b)

    # Set borders of mask to zero to ignore edge effects.
    padded_mask = zero_mask_border(mask, patch_size)
    diff = abs_robust_loss(hamming)
    diff *= padded_mask
    return torch.sum(diff) / (torch.sum(padded_mask.detach()) + 1e-6)


def census_loss_no_penalty(image_a, image_b, mask, patch_size=7):
    """Compares the similarity of the census transform of two images."""
    census_image_a = census_transform(image_a, patch_size)
    census_image_b = census_transform(image_b, patch_size)

    hamming = soft_hamming(census_image_a, census_image_b)

    # Set borders of mask to zero to ignore edge effects.
    padded_mask = zero_mask_border(mask, patch_size)

    return hamming, padded_mask / (torch.sum(padded_mask.detach()) + 1e-6)


def ssim_loss(image_a, image_b, mask, patch_size=7):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(patch_size, 1, patch_size//2)(image_a)
    mu_y = nn.AvgPool2d(patch_size, 1, patch_size//2)(image_b)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(patch_size, 1, patch_size//2)(image_a * image_a) - mu_x_sq
    sigma_y = nn.AvgPool2d(patch_size, 1, patch_size//2)(image_b * image_b) - mu_y_sq
    sigma_xy = nn.AvgPool2d(patch_size, 1, patch_size//2)(image_a * image_b) - mu_x_mu_y

    #SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    #SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    #SSIM = SSIM_n / SSIM_d
    # dist = torch.clamp((1 - SSIM) / 2, 0, 1)

    S1 = (2 * mu_x_mu_y + C1) / (mu_x_sq + mu_y_sq + C1)
    S2 = (2 * sigma_xy + C2) / (sigma_x + sigma_y + C2)
    d1_sq = torch.clamp(1 - S1, min=0, max=1)
    d2_sq = torch.clamp(1 - S2, min=0, max=1)
    
    padded_mask = zero_mask_border(mask, patch_size=patch_size)
    return [d1_sq, d2_sq], padded_mask / (torch.sum(padded_mask.detach()) + 1e-6)


def robust_l1(x):
    return (x + 0.001 ** 2) ** 0.5