import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
from easydict import EasyDict
from torchvision import transforms
from transforms import sep_transforms

from utils.flow_utils import flow_to_image, resize_flow, load_flow, sp_plot
from utils.torch_utils import restore_model
from models.pwclite_prob import PWCLiteProb



class TestHelper():
    def __init__(self, cfg):
        self.cfg = EasyDict(cfg)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu")
        self.model = self.init_model()
        self.input_transform = transforms.Compose([
            sep_transforms.Zoom(*self.cfg.test_shape),
            sep_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ])

    def init_model(self):
        model = PWCLiteProb(self.cfg.model)
        # print('Number fo parameters: {}'.format(model.num_parameters()))
        model = model.to(self.device)
        model = restore_model(model, self.cfg.pretrained_model)
        model.eval()
        return model

    def run(self, imgs):
        imgs = [self.input_transform(img).unsqueeze(0) for img in imgs]
        img_pair = torch.cat(imgs, 1).to(self.device)
        return self.model(img_pair)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='checkpoints/KITTI15/pwclite_ar.tar')
    parser.add_argument('-s', '--test_shape', default=[448, 1024], type=int, nargs=2)
    #parser.add_argument('-i', '--img_list', nargs='+',
    #                    default=['examples/frame_0014.png', 'examples/frame_0015.png'])
    parser.add_argument('-i', '--img_list', nargs='+',
                        default=['examples/frame_0011.png', 'examples/frame_0012.png'])
    #parser.add_argument('-i', '--img_list', nargs='+',
    #                    default=['examples/frame_0001.png', 'examples/frame_0002.png'])
    parser.add_argument('-f', '--flow', default='examples/frame_0011.flo')
    #parser.add_argument('-f', '--flow', default='examples/frame_0001.flo')
    #parser.add_argument('-f', '--flow', default='examples/frame_0014.flo')
    args = parser.parse_args()

    cfg = {
        'model': {
            'upsample': True,
            'n_frames': len(args.img_list),
            'reduce_dense': True
        },
        'pretrained_model': args.model,
        'test_shape': args.test_shape,
    }

    th = TestHelper(cfg)

    imgs = [imageio.imread(img).astype(np.float32) for img in args.img_list]
    h, w = imgs[0].shape[:2]

    # Run inference
    flow_12 = th.run(imgs)['flows_fw'][0]

    # Flow
    flow_12 = resize_flow(flow_12, (h, w))
    np_flow_12 = flow_12[0, 0:2].detach().cpu().numpy().transpose([1, 2, 0])

    # Entropy
    np_entropy_12 = torch.sum(flow_12[0, 2:4], axis=0).detach().cpu().numpy()
    #np_entropy_12 = 1000*np.ones_like(np_entropy_12, dtype=float)

    # Ground-truth
    np_gt_12 = load_flow(args.flow)

    # Error
    epe = np.sqrt(np.sum((np_flow_12 - np_gt_12)**2, axis=2))

    # Sparsification plot
    start = time.time()
    splot = sp_plot(epe, np_entropy_12, n=25, alpha=100.0, eps=5e-2)
    end = time.time()
    print('Elapsed time: ', end - start)
    oplot = sp_plot(epe, epe, n=25, alpha=100.0, eps=5e-2)
    # Cummulate AUC
    frac = np.linspace(0, 1, 25)
    sauc = np.trapz(splot / splot[0], x=frac)
    oauc = np.trapz(oplot / oplot[0], x=frac)

    print('Auc:', sauc)
    print('Auc_oracle:', oauc)
    plt.plot(frac, splot, '+-')
    plt.plot(frac, oplot, '+-')
    plt.show()

    #fig, ax = plt.subplots(2,2)
    #vis_gt = flow_to_image(np_gt_12)
    #vis_flow = flow_to_image(np_flow_12)
    #ax[0,0].imshow(vis_gt)
    #ax[0,1].imshow(vis_flow)
    #ax[1,0].imshow(np_entropy_12)
    #ax[1,1].imshow(np.log(epe))
    #plt.show()

