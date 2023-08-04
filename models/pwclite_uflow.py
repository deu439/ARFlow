import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.warp_utils import flow_warp
#from .correlation_package.correlation import Correlation
from .correlation_native import Correlation


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True)
        )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


def normalize_features(features_list):
    """
    Standardize a list of features across items, and pixels
    """
    features = torch.cat(features_list, dim=1)  # Concatenate over channels
    mean = torch.mean(features, dim=(-3, -2, -1), keepdim=True)
    var = torch.var(features, dim=(-3, -2, -1), keepdim=True)
    std = torch.sqrt(var + 1e-16)
    return [(feature - mean) / std for feature in features_list]

class FeatureExtractor(nn.Module):
    def __init__(self, num_chs):
        super(FeatureExtractor, self).__init__()
        self.num_chs = num_chs
        self.convs = nn.ModuleList()

        # bottom to top
        for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            layer = nn.Sequential(
                conv(ch_in, ch_out, stride=2),
                conv(ch_out, ch_out)
            )
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x)
            feature_pyramid.append(x)

        return feature_pyramid[::-1]


class FlowEstimatorDense(nn.Module):
    def __init__(self, ch_in):
        super(FlowEstimatorDense, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(ch_in + 128, 128)
        self.conv3 = conv(ch_in + 256, 96)
        self.conv4 = conv(ch_in + 352, 64)
        self.conv5 = conv(ch_in + 416, 32)
        self.feat_dim = ch_in + 448
        self.conv_last = conv(ch_in + 448, 2, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out


class FlowEstimatorReduce(nn.Module):
    # can reduce 25% of training time.
    def __init__(self, ch_in):
        super(FlowEstimatorReduce, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(128, 128)
        self.conv3 = conv(128 + 128, 96)
        self.conv4 = conv(128 + 96, 64)
        self.conv5 = conv(96 + 64, 32)
        self.feat_dim = 32
        self.predict_flow = conv(64 + 32, 2, isReLU=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x1, x2], dim=1))
        x4 = self.conv4(torch.cat([x2, x3], dim=1))
        x5 = self.conv5(torch.cat([x3, x4], dim=1))
        flow = self.predict_flow(torch.cat([x4, x5], dim=1))
        return x5, flow


class ContextNetwork(nn.Module):
    def __init__(self, ch_in):
        super(ContextNetwork, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 2),
            conv(128, 128, 3, 1, 4),
            conv(128, 96, 3, 1, 8),
            conv(96, 64, 3, 1, 16),
            conv(64, 32, 3, 1, 1),
            conv(32, 2, isReLU=False)
        )

    def forward(self, x):
        return self.convs(x)


class PWCLiteUflow(nn.Module):
    def __init__(self, cfg):
        super(PWCLiteUflow, self).__init__()

        self.search_range = 4
        # Level numbers: [0, 1, 2, 3, 4, 5]
        self.num_chs = [3, 16, 32, 32, 32, 32]
        self.output_level = 3   # In reverse order
        self.num_levels = 6     # 5 + input layer (images)
        self.deconv_chs = 32    # Number of channels for the deconvolved activations
        self.level_dropout = cfg.level_dropout
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)
        self.feature_norm = cfg.feature_norm

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)

        self.n_frames = cfg.n_frames
        self.reduce_dense = cfg.reduce_dense

        self.corr = Correlation(pad_size=self.search_range, kernel_size=1,
                                max_displacement=self.search_range, stride1=1,
                                stride2=1, corr_multiply=1)

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        # Build flow estimators networks only up to output_level (top to bottom)
        rev_chs = self.num_chs[::-1]
        self.flow_estimators = nn.ModuleList()
        for l, num in enumerate(rev_chs[0:self.output_level+1]):
            num_ch_in = num + (self.dim_corr + 2) * (self.n_frames - 1)
            # The last layer does not take deconvolved activations as input
            if l > 0:
                num_ch_in += self.deconv_chs

            if self.reduce_dense:
                self.flow_estimators.append(FlowEstimatorReduce(num_ch_in))
            else:
                self.flow_estimators.append(FlowEstimatorDense(num_ch_in))

        # Build context network
        self.context_networks = ContextNetwork((self.flow_estimators[self.output_level].feat_dim + 2) * (self.n_frames - 1))

        # Build deconvolution networks only up to output_level - 1 (top to bottom)
        self.deconv_networks = nn.ModuleList()
        for l, estimator in enumerate(self.flow_estimators[0:self.output_level]):
            self.deconv_networks.append(deconv(estimator.feat_dim, self.deconv_chs))

    def num_parameters(self):
        return sum(
            [p.data.nelement() if p.requires_grad else 0 for p in self.parameters()])

    def init_weights(self):
        for layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                #nn.init.kaiming_normal_(layer.weight.data, mode='fan_in')
                torch.nn.init.xavier_uniform(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.ConvTranspose2d):
                #nn.init.kaiming_normal_(layer.weight.data, mode='fan_in')
                torch.nn.init.xavier_uniform(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward_2_frames(self, x1_pyramid, x2_pyramid):
        # outputs
        flows = []

        # init
        b_size, _, h_x1, w_x1, = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        flow = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()

        # Iterate top to bottom
        for l, (x1, x2) in enumerate(zip(x1_pyramid[0:self.output_level+1], x2_pyramid[0:self.output_level+1])):
            # warping
            if l == 0:
                x2_warp = x2
            else:
                flow = F.interpolate(flow * 2, scale_factor=2, mode='bilinear', align_corners=True)
                x2_warp = flow_warp(x2, flow)

            # correlation volume
            if self.feature_norm:
                x1, x2_warp = normalize_features([x1, x2_warp])
            out_corr = self.corr(x1, x2_warp)
            out_corr_relu = self.leakyRELU(out_corr)

            # concat and estimate flow
            if l == 0:
                act, flow_res = self.flow_estimators[l](torch.cat([out_corr_relu, x1, flow], dim=1))
            else:
                act_deconv = self.deconv_networks[l-1](act)
                act, flow_res = self.flow_estimators[l](torch.cat([out_corr_relu, x1, flow, act_deconv], dim=1))

            # level dropout
            if self.training and self.level_dropout > 0:
                drop = (torch.rand(1) > self.level_dropout).float().item()
                flow_res = flow_res * drop
                act = act * drop

            # Residual connection
            flow = flow + flow_res

            # Store the flow
            flows.append(flow)

        # Context network refinement at the output level
        flow_fine = self.context_networks(torch.cat([act, flow], dim=1))
        # Level dropout
        if self.training and self.level_dropout > 0:
            drop = (torch.rand(1) > self.level_dropout).float().item()
            flow_fine = flow_fine * drop
        flow = flow + flow_fine
        flows[-1] = flow

        # Append upsampled flow
        upsampled_flow = F.interpolate(flow * 4, scale_factor=4, mode='bilinear', align_corners=True)
        flows.append(upsampled_flow)

        return flows[::-1]

    def forward(self, x, with_bk=False):
        n_frames = x.size(1) / 3

        imgs = [x[:, 3 * i: 3 * i + 3] for i in range(int(n_frames))]
        x = [self.feature_pyramid_extractor(img) + [img] for img in imgs]

        res_dict = {}
        if n_frames == 2:
            res_dict['flows_fw'] = self.forward_2_frames(x[0], x[1])
            if with_bk:
                res_dict['flows_bw'] = self.forward_2_frames(x[1], x[0])
        else:
            raise NotImplementedError
        return res_dict

