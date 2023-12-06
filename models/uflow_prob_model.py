import torch.nn as nn
import torch
import torch.nn.functional as func
import math

import utils.uflow_utils as uflow_utils


def normalize_features(feature_list, normalize, center, moments_across_channels,
                       moments_across_images):
    """Normalizes feature tensors (e.g., before computing the cost volume).

    Args:
      feature_list: list of tf.tensors, each with dimensions [b, c, h, w]
      normalize: bool flag, divide features by their standard deviation
      center: bool flag, subtract feature mean
      moments_across_channels: bool flag, compute mean and std across channels
      moments_across_images: bool flag, compute mean and std across images

    Returns:
      list, normalized feature_list
    """

    # Compute feature statistics.

    dim = [1, 2, 3] if moments_across_channels else [2, 3]

    means = []
    vars = []

    for feature_image in feature_list:
        mean = torch.mean(feature_image, dim=dim, keepdim=True)
        var = torch.var(feature_image, dim=dim, keepdim=True)
        means.append(mean)
        vars.append(var)

    if moments_across_images:
        means = [torch.mean(torch.stack(means), dim=0, keepdim=False)] * len(means)
        vars = [torch.mean(torch.stack(vars), dim=0, keepdim=False)] * len(vars)

    stds = [torch.sqrt(v + 1e-16) for v in vars]

    # Center and normalize features.
    if center:
        feature_list = [
            f - mean for f, mean in zip(feature_list, means)
        ]
    if normalize:
        feature_list = [f / std for f, std in zip(feature_list, stds)]

    return feature_list


def compute_cost_volume(features1, features2, max_displacement):
    """Compute the cost volume between features1 and features2.

    Displace features2 up to max_displacement in any direction and compute the
    per pixel cost of features1 and the displaced features2.

    Args:
      features1: tf.tensor of shape [b, h, w, c]
      features2: tf.tensor of shape [b, h, w, c]
      max_displacement: int, maximum displacement for cost volume computation.

    Returns:
      tf.tensor of shape [b, h, w, (2 * max_displacement + 1) ** 2] of costs for
      all displacements.
    """

    # Set maximum displacement and compute the number of image shifts.
    _, _, height, width = features1.shape
    if max_displacement <= 0 or max_displacement >= height:
        raise ValueError(f'Max displacement of {max_displacement} is too large.')

    max_disp = max_displacement
    num_shifts = 2 * max_disp + 1

    # Pad features2 and shift it while keeping features1 fixed to compute the
    # cost volume through correlation.

    # Pad features2 such that shifts do not go out of bounds.
    features2_padded = torch.nn.functional.pad(
        input=features2,
        pad=[max_disp, max_disp, max_disp, max_disp],
        mode='constant')
    cost_list = []
    for i in range(num_shifts):
        for j in range(num_shifts):
            prod = features1 * features2_padded[:, :, i:(height + i), j:(width + j)]
            corr = torch.mean(prod, dim=1, keepdim=True)
            cost_list.append(corr)
    cost_volume = torch.cat(cost_list, dim=1)
    return cost_volume


class PWCProbFlow(nn.Module):
    def __init__(self, cfg):
        super(PWCProbFlow, self).__init__()

        self._leaky_relu_alpha = 0.1
        self._drop_out_rate = cfg.level_dropout
        self._num_context_up_channels = 32
        self._num_levels = 5
        self._normalize_before_cost_volume = cfg.feature_norm
        self._channel_multiplier = 1
        self._use_cost_volume = True
        self._use_feature_warp = True
        self._accumulate_flow = True
        self._shared_flow_decoder = False
        self._align_corners = cfg.align_corners
        self._out_channels = cfg.out_channels
        # There's no more than 4 channels at intermediate levels, because upsampling off-diagonal elements of covariance
        # or precision matrix does not make sense.
        self._out_channels_int = 4
        # Bias added to the diagonal elements of covariance/precision matrix when upsampling.
        self._diag_bias = -math.log(2) if cfg.inv_cov else math.log(2)
        self._inv_cov = cfg.inv_cov

        self._refine_model = self._build_refinement_model()
        self._flow_layers = self._build_flow_layers()
        if not self._use_cost_volume:
            self._cost_volume_surrogate_convs = self._build_cost_volume_surrogate_convs()
        if self._num_context_up_channels:
            self._context_up_layers = self._build_upsample_layers(
                num_channels=int(self._num_context_up_channels * self._channel_multiplier))
        if self._shared_flow_decoder:
            # pylint:disable=invalid-name
            self._1x1_shared_decoder = self._build_1x1_shared_decoder()

        self._feature_pyramid_extractor = PWCFeaturePyramid()

    def init_weights(self):
        for layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight.data, mode='fan_in')
                #torch.nn.init.xavier_uniform(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight.data, mode='fan_in')
                #torch.nn.init.xavier_uniform(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def upsample_out(self, out):
        channels = out.shape[1]
        if channels > 4:
            flow, log_diag, rest = torch.split(out, [2, 2, channels - 4], dim=1)
            flow_up = uflow_utils.upsample(flow, is_flow=True, align_corners=self._align_corners)
            log_diag_up = uflow_utils.upsample(log_diag + self._diag_bias, is_flow=False,
                                               align_corners=self._align_corners)
            rest_up = uflow_utils.upsample(rest, is_flow=False, align_corners=self._align_corners)
            out_up = torch.cat([flow_up, log_diag_up, rest_up], dim=1)
        else:
            flow, log_diag = torch.split(out, [2, 2], dim=1)
            flow_up = uflow_utils.upsample(flow, is_flow=True, align_corners=self._align_corners)
            log_diag_up = uflow_utils.upsample(log_diag + self._diag_bias, is_flow=False,
                                               align_corners=self._align_corners)
            out_up = torch.cat([flow_up, log_diag_up], dim=1)

        return out_up

    def forward_2_frames(self, feature_pyramid1, feature_pyramid2):
        """Run the model."""
        context = None
        context_up = None
        out_up = None
        outs = []

        # Go top down through the levels to the second to last one to estimate flow.
        for level, (features1, features2) in reversed(
                list(enumerate(zip(feature_pyramid1, feature_pyramid2)))[1:]):

            # Initialize for coarsest level
            if out_up is None:
                batch_size, _, height, width = list(features1.shape)
                flow_up = torch.zeros([batch_size, 2, height, width], device=features1.device)
                # Start at log_diag ~ 0.0 at level 2 (the output level)
                log_diag_up = -(self._num_levels-2) * self._diag_bias \
                              * torch.ones([batch_size, 2, height, width], device=features1.device)
                out_up = torch.cat([flow_up, log_diag_up], dim=1)

                if self._num_context_up_channels:
                    num_channels = int(self._num_context_up_channels * self._channel_multiplier)
                    context_up = torch.zeros([batch_size, num_channels, height, width]).to(features1.device)

            # Warp features2 with upsampled flow from higher level.
            if out_up is None or not self._use_feature_warp:
                warped2 = features2
            else:
                warp_up = uflow_utils.flow_to_warp(out_up[:, 0:2])
                warped2 = uflow_utils.resample2(features2, warp_up)

            # Compute cost volume by comparing features1 and warped features2.
            features1_normalized, warped2_normalized = normalize_features(
                [features1, warped2],
                normalize=self._normalize_before_cost_volume,
                center=self._normalize_before_cost_volume,
                moments_across_channels=True,
                moments_across_images=True)

            if self._use_cost_volume:
                cost_volume = compute_cost_volume(features1_normalized, warped2_normalized, max_displacement=4)
            else:
                concat_features = torch.cat([features1_normalized, warped2_normalized], dim=1)
                cost_volume = self._cost_volume_surrogate_convs[level](concat_features)

            cost_volume = func.leaky_relu(cost_volume, negative_slope=self._leaky_relu_alpha)

            if self._shared_flow_decoder:
                # This will ensure to work for arbitrary feature sizes per level.
                conv_1x1 = self._1x1_shared_decoder[level]
                features1 = conv_1x1(features1)

            # Compute context and flow from previous flow, cost volume, and features1.
            x_in = torch.cat([cost_volume, features1], dim=1)
            if out_up is not None:
                x_in = torch.cat([out_up, x_in], dim=1)
            if context_up is not None:
                x_in = torch.cat([context_up, x_in], dim=1)

            # Use dense-net connections.
            x_out = None
            if self._shared_flow_decoder:
                # reuse the same flow decoder on all levels
                flow_layers = self._flow_layers
            else:
                flow_layers = self._flow_layers[level]
            for layer in flow_layers[:-1]:
                x_out = layer(x_in)
                x_in = torch.cat([x_in, x_out], dim=1)
            context = x_out
            out = flow_layers[-1](context)

            # Level dropout
            if self.training and self._drop_out_rate > 0:
                maybe_dropout = (torch.rand(1) > self._drop_out_rate).float().item()
                context = context * maybe_dropout
                out = out * maybe_dropout

            # Pad channels if needed
            if out.shape[1] > self._out_channels_int:
                shape = list(out_up.shape)
                shape[1] = self._out_channels - out_up.shape[1]
                padding = torch.zeros(shape).type_as(out_up)
                out_up = torch.cat((out_up, padding), 1)

            if out_up is not None and self._accumulate_flow:
                out = out + out_up

            # Upsample for the next lower level - treat the flow channels separately!
            out_up = self.upsample_out(out)

            if self._num_context_up_channels:
                context_up = self._context_up_layers[level](context)

            # Append results to list.
            outs.insert(0, out)

        # Pad channels if needed
        if out.shape[1] < self._out_channels:
            shape = list(out.shape)
            shape[1] = self._out_channels - out.shape[1]
            padding = torch.zeros(shape).type_as(out)
            out = torch.cat((out, padding), 1)

        # Refine flow at level 2
        refinement = torch.cat([context, out], dim=1)
        for layer in self._refine_model:
            refinement = layer(refinement)

        # Level dropout
        if self.training and self._drop_out_rate > 0:
            maybe_dropout = (torch.rand(1) > self._drop_out_rate).float().item()
            # note that operation must not be inplace, otherwise autograd will fail pathetically
            refinement = refinement * maybe_dropout

        refined_out = out + refinement

        flow, log_diag, rest = torch.split(refined_out, [2, 2, self._out_channels - 4], dim=1)
        # Ensure log(precision)/2 is not too small
        if self._inv_cov:
            outs[0] = torch.cat([flow, torch.clamp(log_diag, min=-5.0), rest], dim=1)
        # Ensure log(variance)/2 is not too large
        else:
            outs[0] = torch.cat([flow, torch.clamp(log_diag, max=10.0), rest], dim=1)

        # Upsample 4x up to the 0-th level
        out_1 = self.upsample_out(outs[0])
        out_0 = self.upsample_out(out_1)
        outs.insert(0, out_1)
        outs.insert(0, out_0)

        return outs

    def forward(self, x, with_bk=True):
        n_frames = x.size(1) / 3

        imgs = [x[:, 3 * i: 3 * i + 3] for i in range(int(n_frames))]
        x = [self._feature_pyramid_extractor(img) for img in imgs]

        res_dict = {}
        if n_frames == 2:
            res_dict['flows_fw'] = self.forward_2_frames(x[0], x[1])
            if with_bk:
                res_dict['flows_bw'] = self.forward_2_frames(x[1], x[0])
        else:
            raise NotImplementedError
        return res_dict

    def _build_cost_volume_surrogate_convs(self):
        layers = nn.ModuleList()
        for _ in range(self._num_levels):
            layers.append(nn.Sequential(
                #nn.ZeroPad2d((2,1,2,1)), # should correspond to "SAME" in keras
                nn.Conv2d(
                    in_channels=int(64 * self._channel_multiplier),
                    out_channels=int(64 * self._channel_multiplier),
                    kernel_size=(4, 4),
                    padding='same'))
            )
        return layers

    def _build_upsample_layers(self, num_channels):
        """Build layers for upsampling via deconvolution."""
        layers = []
        for unused_level in range(self._num_levels):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=(4, 4),
                    stride=2,
                    padding=1))
        return nn.ModuleList(layers)

    def _build_flow_layers(self):
        """Build layers for flow estimation."""
        # Empty list of layers level 0 because flow is only estimated at levels > 0.
        result = nn.ModuleList([None])

        block_layers = [128, 128, 96, 64, 32]

        for i in range(1, self._num_levels):
            layers = nn.ModuleList()
            last_in_channels = (64+32) if not self._use_cost_volume else (81+32)
            # In contrast to UFlow we feed zero-flow and constant variance/precision at the fifth level input
            #if i != self._num_levels-1:
            last_in_channels += self._out_channels_int + self._num_context_up_channels * self._channel_multiplier

            for c in block_layers:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=last_in_channels,
                            out_channels=int(c * self._channel_multiplier),
                            kernel_size=(3, 3),
                            stride=1,
                            padding='same'),
                        nn.LeakyReLU(
                            negative_slope=self._leaky_relu_alpha)
                    ))
                last_in_channels += int(c * self._channel_multiplier)
            layers.append(
                nn.Conv2d(
                    in_channels=int(block_layers[-1] * self._channel_multiplier),
                    out_channels=self._out_channels if i == 1 else self._out_channels_int,
                    kernel_size=(3, 3),
                    padding='same'))
            if self._shared_flow_decoder:
                return layers
            result.append(layers)
        return result

    def _build_refinement_model(self):
        """Build model for flow refinement using dilated convolutions."""
        layers = []
        last_in_channels = 32*self._channel_multiplier+self._out_channels
        for c, d in [(128, 1), (128, 2), (128, 4), (96, 8), (64, 16), (32, 1)]:
            layers.append(
                nn.Conv2d(
                    in_channels=last_in_channels,
                    out_channels=int(c * self._channel_multiplier),
                    kernel_size=(3, 3),
                    stride=1,
                    padding='same',
                    dilation=d))
            layers.append(
                nn.LeakyReLU(negative_slope=self._leaky_relu_alpha))
            last_in_channels = int(c * self._channel_multiplier)
        layers.append(
            nn.Conv2d(
                in_channels=last_in_channels,
                out_channels=self._out_channels,
                kernel_size=(3, 3),
                stride=1,
                padding='same'))
        return nn.ModuleList(layers)

    def _build_1x1_shared_decoder(self):
        """Build layers for flow estimation."""
        # Empty list of layers level 0 because flow is only estimated at levels > 0.
        result = nn.ModuleList([nn.ModuleList()])
        for _ in range(1, self._num_levels):
            result.append(
                nn.Conv2d(
                    in_channels= 32,
                    out_channels=32,
                    kernel_size=(1, 1),
                    stride=1,
                    padding='same'))
        return result


class PWCFeaturePyramid(nn.Module):
  """Model for computing a feature pyramid from an image."""

  def __init__(self,
               leaky_relu_alpha=0.1,
               filters=None,
               level1_num_layers=3,
               level1_num_filters=32,
               level1_num_1x1=0,
               original_layer_sizes=False,
               num_levels=5,
               channel_multiplier=1.,
               pyramid_resolution='half',
               num_channels=3):
    """Constructor.

    Args:
      leaky_relu_alpha: Float. Alpha for leaky ReLU.
      filters: Tuple of tuples. Used to construct feature pyramid. Each tuple is
        of form (num_convs_per_group, num_filters_per_conv).
      level1_num_layers: How many layers and filters to use on the first
        pyramid. Only relevant if filters is None and original_layer_sizes
        is False.
      level1_num_filters: int, how many filters to include on pyramid layer 1.
        Only relevant if filters is None and original_layer_sizes if False.
      level1_num_1x1: How many 1x1 convolutions to use on the first pyramid
        level.
      original_layer_sizes: bool, if True, use the original PWC net number
        of layers and filters.
      num_levels: int, How many feature pyramid levels to construct.
      channel_multiplier: float, used to scale up or down the amount of
        computation by increasing or decreasing the number of channels
        by this factor.
      pyramid_resolution: str, specifies the resolution of the lowest (closest
        to input pyramid resolution)
      use_bfloat16: bool, whether or not to run in bfloat16 mode.
    """

    super(PWCFeaturePyramid, self).__init__()

    self._channel_multiplier = channel_multiplier
    if num_levels > 6:
      raise NotImplementedError('Max number of pyramid levels is 6')
    if filters is None:
      if original_layer_sizes:
        # Orig - last layer
        filters = ((3, 16), (3, 32), (3, 64), (3, 96), (3, 128),
                   (3, 196))[:num_levels]
      else:
        filters = ((level1_num_layers, level1_num_filters), (3, 32), (3, 32),
                   (3, 32), (3, 32), (3, 32))[:num_levels]
    assert filters
    assert all(len(t) == 2 for t in filters)
    assert all(t[0] > 0 for t in filters)

    self._leaky_relu_alpha = leaky_relu_alpha
    self._convs = nn.ModuleList()
    self._level1_num_1x1 = level1_num_1x1

    c = num_channels

    for level, (num_layers, num_filters) in enumerate(filters):
      group = nn.ModuleList()
      for i in range(num_layers):
        stride = 1
        if i == 0 or (i == 1 and level == 0 and
                      pyramid_resolution == 'quarter'):
          stride = 2
        conv = nn.Conv2d(
            in_channels=c,
            out_channels=int(num_filters * self._channel_multiplier),
            kernel_size=(3,3) if level > 0 or i < num_layers - level1_num_1x1 else (1, 1),
            stride=stride,
            padding='valid')
        group.append(conv)
        c = int(num_filters * self._channel_multiplier)
      self._convs.append(group)

  def forward(self, x, split_features_by_sample=False):
    x = x * 2. - 1.  # Rescale input from [0,1] to [-1, 1]
    features = []
    for level, conv_tuple in enumerate(self._convs):
      for i, conv in enumerate(conv_tuple):
        if level > 0 or i < len(conv_tuple) - self._level1_num_1x1:
          x = func.pad(
              x,
              pad=[1, 1, 1, 1],
              mode='constant',
              value=0)
        x = conv(x)
        x = func.leaky_relu(x, negative_slope=self._leaky_relu_alpha)
      features.append(x)

    if split_features_by_sample:

      # Split the list of features per level (for all samples) into a nested
      # list that can be indexed by [sample][level].

      n = len(features[0])
      features = [[f[i:i + 1] for f in features] for i in range(n)]  # pylint: disable=g-complex-comprehension

    return features
