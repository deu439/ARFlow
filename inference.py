import json
import argparse
import pprint
import time
import numpy as np
import cv2
import math
from path import Path

import torch
from easydict import EasyDict

from utils.torch_utils import restore_model
from utils.misc_utils import AverageMeter, mixture_entropy
from utils.flow_utils import write_flow
from datasets.get_dataset import get_dataset
from models.get_model import get_model


class TestHelper():
    def __init__(self, cfg, data_loader, model):
        self.cfg = cfg
        self.data_loader = data_loader
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self._init_model(model)

    def _init_model(self, model):
        # print('Number fo parameters: {}'.format(model.num_parameters()))
        model = model.to(self.device)
        model = restore_model(model, self.cfg.inference.pretrained_model, device=self.device)
        model.eval()
        return model

    def run(self):
        # Meter for batch time measurement
        batch_time = AverageMeter()

        # Iterate through datasets
        for (i_set, loader), dataset_cfg in zip(enumerate(self.data_loader), self.cfg.data):

            # Iterate through samples
            for i_step, data in enumerate(loader):
                # Measure time batch time
                end = time.time()

                # Get data
                img1, img2 = data['img1'].to(self.device), data['img2'].to(self.device)

                # Compute prediction
                res_dict = self.model(img1, img2)
                flows = res_dict['flows_fw']
                pred_flow_np = flows[0][:, 0:2].detach().cpu().numpy().transpose([0, 2, 3, 1])

                # Compute entropy of the horizontal and vertical component.
                if self.cfg.loss.approx == 'diag':
                    uv_entropy = flows[0][:, 2:4]

                elif self.cfg.loss.approx == 'mixture':
                    K = self.cfg.loss.n_components
                    mean = flows[0][:, 0:K * 2]
                    logstd = flows[0][:, K * 2:K * 2 + 2]
                    uv_entropy = mixture_entropy(mean, logstd, n_samples=100)

                elif self.cfg.loss.approx == 'sparse':
                    if self.cfg.loss.inv_cov:
                        log_diag = flows[2][:, 2:4]
                        left = flows[2][:, 4:6, :, :-1]
                        over = flows[2][:, 6:8, :-1, :]
                        uv_entropy = inverse_diagonal(torch.exp(log_diag).contiguous(), left.contiguous(),
                                                      over.contiguous())
                        uv_entropy = uflow_utils.upsample(uv_entropy + 2 * math.log(4), scale_factor=4, is_flow=False,
                                                          align_corners=False)
                    else:
                        uv_entropy = flows[0][:, 2:4]

                elif self.cfg.loss.approx == 'lowrank':
                    std = flows[2][:, 2:2 + 2 * self.loss_func.cfg.columns]
                    std_u = std[:, 0::2]
                    std_v = std[:, 1::2]
                    u_entropy = torch.log(torch.sum(std_u * std_u, dim=1, keepdim=True)) / 2
                    v_entropy = torch.log(torch.sum(std_v * std_v, dim=1, keepdim=True)) / 2
                    uv_entropy = torch.concat((u_entropy, v_entropy), dim=1)
                    uv_entropy = uflow_utils.upsample(uv_entropy + 2 * math.log(4), scale_factor=4, is_flow=False,
                                                      align_corners=False)
                else:
                    raise NotImplementedError(f"Invalid approximation {self.loss_func.cfg.approx}!")

                pred_entropy_np = uv_entropy.detach().cpu().numpy().transpose([0, 2, 3, 1])

                # Scale to the original size
                for pred_flow, pred_entropy, img1_orgsize, img1_rpath in zip(pred_flow_np, pred_entropy_np, data['img1_orgsize'], data['img1_rpath']):
                    # img1_orgsize (original image size) can be different from img1.size() !
                    print(img1_orgsize.squeeze()[1:])
                    img1_orgsize = img1_orgsize.squeeze()
                    H = img1_orgsize[1].item()
                    W = img1_orgsize[2].item()

                    # Resample flow - back to original shape
                    h, w = pred_flow.shape[:2]
                    pred_flow[:, :, 0] = pred_flow[:, :, 0] / w * W
                    pred_flow[:, :, 1] = pred_flow[:, :, 1] / h * H
                    pred_flow = cv2.resize(pred_flow, (W, H), interpolation=cv2.INTER_LINEAR)

                    # Resample entropy - back to original shape
                    pred_entropy[:, :, 0] = pred_entropy[:, :, 0] - 2 * math.log(w) + 2 * math.log(W)
                    pred_entropy[:, :, 1] = pred_entropy[:, :, 1] - 2 * math.log(h) + 2 * math.log(H)
                    pred_entropy = cv2.resize(pred_entropy, (W, H), interpolation=cv2.INTER_LINEAR)

                    # Store the result
                    flow_path = Path(dataset_cfg.out_root) / Path(img1_rpath).with_suffix('.flo')
                    entropy_path = Path(dataset_cfg.out_root) / Path(img1_rpath).with_suffix('.npy')
                    flow_path.parent.makedirs_p()   # Ensure the target directory exists
                    write_flow(flow_path, pred_flow)
                    np.save(entropy_path, pred_entropy)

                # measure elapsed time
                batch_time.update(time.time() - end)
                print(batch_time, i_set, i_step)
                print('Inference: {0}[{1}/{2}]\t Time {3}\t '.format(i_set, i_step, len(loader.dataset), batch_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    parser.add_argument('-m', '--model', default=None)
    args = parser.parse_args()

    # Read the configuration
    with open(args.config) as f:
        cfg = EasyDict(json.load(f))

    print("=> fetching img pairs.")
    _, valid_set = get_dataset(cfg)

    valid_len = sum([len(s) for s in valid_set])
    print('{} samples found'.format(valid_len))

    # Default validation batch size is 1 for compatibility with KITTI dataset
    valid_batch_size = cfg.inference.get('valid_batch_size', 1)
    valid_loader = [torch.utils.data.DataLoader(
        s, batch_size=valid_batch_size,
        num_workers=min(4, cfg.inference.workers),
        pin_memory=True, shuffle=False) for s in valid_set]

    # Use a different model checkpoint if specified on commandline
    if args.model is not None:
        cfg.train.pretrained_model = args.model

    # show configurations
    cfg_str = pprint.pformat(cfg)
    print('=> configurations \n ' + cfg_str)

    # Instantiate model
    model = get_model(cfg.model)

    # Run inference
    inf = TestHelper(cfg, valid_loader, model)
    inf.run()