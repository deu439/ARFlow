import torch
from path import Path
from abc import abstractmethod, ABCMeta

import numpy as np
from PIL import Image
from torchvision import transforms

from torch.utils.data import Dataset
from utils.flow_utils import load_flow


def image_to_tensor(img):
    """
    Converts a PIL Image (H x W x C) to a [0,1] normalized torch.FloatTensor of shape (C x H x W).
    """

    img = transforms.functional.pil_to_tensor(img)
    return img.float() / 255.0


def flow_to_tensor(flow):
    """
    Converts numpy flow array (H, W, C) to torch.FloatTensor of shape (C x H x W)
    """
    flow = torch.from_numpy(flow)
    return flow.permute((2, 0, 1)).float()


class ImgSeqDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, root, n_frames, geometric_transform=None, photometric_transform=None):
        self.root = Path(root)
        self.n_frames = n_frames
        self.geometric_transform = geometric_transform
        self.photometric_transform = photometric_transform
        self.samples = self.collect_samples()

    @abstractmethod
    def collect_samples(self):
        pass

    def _load_sample(self, s):
        images = s['imgs']
        images = torch.stack([image_to_tensor(Image.open(self.root / p)) for p in images])

        target = {}
        if 'flow' in s:
            target['flow'] = flow_to_tensor(load_flow(self.root / s['flow']))
        if 'flow_occ' and 'flow_noc' in s:
            flow_occ = flow_to_tensor(load_flow(self.root / s['flow_occ']))
            flow_noc = flow_to_tensor(load_flow(self.root / s['flow_noc']))
            target['flow'] = torch.concat([flow_occ, flow_noc[[2]]], dim=0)
        if 'mask' in s:
            # 0~255 HxWx1
            mask = image_to_tensor(Image.open(self.root / s['mask']))
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            target['mask'] = np.expand_dims(mask, -1)
        if 'flow_bw' in s:
            target['flow_bw'] = flow_to_tensor(load_flow(self.root / s['flow_bw']))
        return images, target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        images, target = self._load_sample(self.samples[idx])

        if self.geometric_transform is not None:
            images = self.geometric_transform(images)

        data = {'img{}'.format(i + 1): img for i, img in enumerate(images)}

        if self.photometric_transform is not None:
            images_ph = self.photometric_transform(images)
            data.update({'img{}_ph'.format(i + 1): img_ph for i, img_ph in enumerate(images_ph)})

        data['target'] = target

        return data


class SintelRaw(ImgSeqDataset):
    def __init__(self, root, n_frames=2, geometric_transform=None, photometric_transform=None):
        super(SintelRaw, self).__init__(root, n_frames, geometric_transform=geometric_transform,
                                        photometric_transform=photometric_transform)

    def collect_samples(self):
        scene_list = self.root.dirs()
        samples = []
        for scene in scene_list:
            img_list = scene.files('*.png')
            img_list.sort()

            for st in range(0, len(img_list) - self.n_frames + 1):
                seq = img_list[st:st + self.n_frames]
                sample = {'imgs': [self.root.relpathto(file) for file in seq]}
                samples.append(sample)
        return samples


class Sintel(ImgSeqDataset):
    def __init__(self, root, n_frames=2, type='clean',
                 subsplit='trainval', with_flow=True, geometric_transform=None, photometric_transform=None):
        self.dataset_type = type
        self.with_flow = with_flow

        self.subsplit = subsplit
        self.training_scene = ['alley_1', 'ambush_4', 'ambush_6', 'ambush_7', 'bamboo_2',
                               'bandage_2', 'cave_2', 'market_2', 'market_5', 'shaman_2',
                               'sleeping_2', 'temple_3']  # Unofficial train-val split

        super(Sintel, self).__init__(root, n_frames, geometric_transform=geometric_transform,
                                     photometric_transform=photometric_transform)

    def collect_samples(self):
        img_dir = self.root / Path(self.dataset_type)
        flow_dir = self.root / 'flow'

        assert img_dir.is_dir()
        assert flow_dir.is_dir() or not self.with_flow

        samples = []
        for flow_map in sorted(img_dir.glob('*/*.png')):
            info = flow_map.splitall()
            scene, filename = info[-2:]
            fid = int(filename[-8:-4])
            if self.subsplit != 'trainval':
                if self.subsplit == 'train' and scene not in self.training_scene:
                    continue
                if self.subsplit == 'val' and scene in self.training_scene:
                    continue

            s = {'imgs': [img_dir / scene / 'frame_{:04d}.png'.format(fid + i) for i in
                          range(self.n_frames)]}
            try:
                assert all([p.is_file() for p in s['imgs']])

                if self.with_flow:
                    if self.n_frames == 3:
                        # for img1 img2 img3, only flow_23 will be evaluated
                        s['flow'] = flow_dir / scene / 'frame_{:04d}.flo'.format(fid + 1)
                    elif self.n_frames == 2:
                        # for img1 img2, flow_12 will be evaluated
                        s['flow'] = flow_dir / scene / 'frame_{:04d}.flo'.format(fid)
                    else:
                        raise NotImplementedError(
                            'n_frames {} with flow or mask'.format(self.n_frames))

                    if self.with_flow:
                        assert s['flow'].is_file()
            except AssertionError:
                print('Incomplete sample for: {}'.format(s['imgs'][0]))
                continue
            samples.append(s)

        return samples


class Chairs2(ImgSeqDataset):
    def __init__(self, root, n_frames=2, split='training', with_flow=True, geometric_transform=None,
                 photometric_transform=None):
        self.with_flow = with_flow
        self.split = split

        root = Path(root)
        super(Chairs2, self).__init__(root, n_frames, geometric_transform=geometric_transform,
                                      photometric_transform=photometric_transform)

    def collect_samples(self):

        if self.n_frames > 2:
            raise NotImplementedError('n_frames {}'.format(self.n_frames))

        samples = []
        if self.split == 'training':
            path = self.root / Path('train')
        else:
            path = self.root / Path('val')

        for flow_map in sorted(path.glob('*flow_01.flo')):
            info = flow_map.splitall()
            filename = info[-1]
            fid = int(filename[0:7])

            # Images
            s = {'imgs': [path / '{:07d}-img_{:d}.png'.format(fid, i) for i in range(self.n_frames)]}
            assert all([p.is_file() for p in s['imgs']])

            if self.with_flow:
                # for img1 img2, flow_12 will be evaluated
                s['flow'] = flow_map
                assert s['flow'].is_file()

                s['flow_bw'] = path / '{:07d}-flow_10.flo'.format(fid)
                assert s['flow_bw'].is_file()

            samples.append(s)

        return samples


class Chairs(ImgSeqDataset):
    def __init__(self, root, n_frames=2, split='training', with_flow=True, geometric_transform=None,
                 photometric_transform=None):
        self.with_flow = with_flow
        self.split = split
        self.valid_indices = [
            6, 18, 43, 46, 59, 63, 97, 112, 118, 121, 122, 132, 133, 153, 161, 249, 264, 265, 292, 294, 296, 300, 317,
            321, 337, 338, 344, 359, 400, 402, 430, 439, 469, 477, 495, 510, 529, 532, 573, 582, 584, 589, 594, 682,
            689, 697, 715, 768, 787, 811, 826, 837, 842, 884, 918, 938, 943, 971, 975, 981, 1017, 1044, 1065, 1119,
            1122, 1134, 1154, 1156, 1159, 1160, 1174, 1188, 1220, 1238, 1239, 1260, 1267, 1279, 1297, 1355, 1379, 1388,
            1495, 1509, 1519, 1575, 1602, 1615, 1669, 1674, 1700, 1713, 1715, 1738, 1842, 1873, 1880, 1902, 1922, 1935,
            1962, 1968, 1979, 2019, 2031, 2040, 2044, 2062, 2114, 2205, 2217, 2237, 2251, 2275, 2293, 2311, 2343, 2360,
            2375, 2383, 2400, 2416, 2420, 2484, 2503, 2505, 2577, 2590, 2591, 2623, 2625, 2637, 2652, 2656, 2659, 2660,
            2665, 2673, 2707, 2708, 2710, 2726, 2733, 2762, 2828, 2865, 2867, 2906, 2923, 2930, 2967, 2973, 2994, 3011,
            3026, 3032, 3041, 3042, 3071, 3114, 3125, 3130, 3138, 3142, 3158, 3184, 3207, 3220, 3248, 3254, 3273, 3277,
            3322, 3329, 3334, 3339, 3342, 3347, 3352, 3397, 3420, 3431, 3434, 3449, 3456, 3464, 3504, 3527, 3530, 3538,
            3556, 3578, 3585, 3592, 3595, 3598, 3604, 3614, 3616, 3671, 3677, 3679, 3698, 3724, 3729, 3735, 3746, 3751,
            3753, 3780, 3783, 3814, 3818, 3820, 3855, 3886, 3945, 3948, 3971, 3986, 4012, 4023, 4072, 4076, 4133, 4159,
            4168, 4191, 4195, 4208, 4247, 4250, 4299, 4308, 4318, 4319, 4320, 4321, 4383, 4400, 4402, 4408, 4417, 4424,
            4485, 4492, 4494, 4518, 4526, 4539, 4579, 4607, 4610, 4621, 4624, 4638, 4647, 4663, 4669, 4717, 4740, 4748,
            4771, 4775, 4777, 4786, 4801, 4846, 4864, 4892, 4905, 4923, 4926, 4957, 4964, 4965, 4995, 5012, 5020, 5037,
            5039, 5042, 5056, 5119, 5123, 5131, 5163, 5165, 5179, 5197, 5228, 5267, 5271, 5274, 5280, 5300, 5311, 5315,
            5364, 5376, 5385, 5394, 5415, 5418, 5434, 5449, 5495, 5506, 5510, 5526, 5567, 5582, 5603, 5610, 5621, 5654,
            5671, 5679, 5691, 5701, 5704, 5725, 5753, 5766, 5804, 5812, 5861, 5882, 5896, 5913, 5916, 5941, 5953, 5967,
            5978, 5989, 6008, 6038, 6062, 6070, 6081, 6112, 6128, 6147, 6162, 6167, 6169, 6179, 6183, 6191, 6221, 6236,
            6254, 6271, 6344, 6373, 6380, 6411, 6412, 6443, 6454, 6482, 6499, 6501, 6510, 6533, 6542, 6544, 6561, 6577,
            6581, 6595, 6596, 6610, 6626, 6630, 6645, 6659, 6674, 6681, 6699, 6700, 6703, 6706, 6742, 6760, 6786, 6793,
            6795, 6810, 6811, 6831, 6839, 6870, 6872, 6890, 6926, 6996, 7004, 7027, 7030, 7081, 7083, 7098, 7103, 7117,
            7166, 7201, 7233, 7272, 7283, 7325, 7334, 7336, 7373, 7388, 7408, 7473, 7475, 7483, 7490, 7500, 7517, 7534,
            7537, 7567, 7621, 7655, 7692, 7705, 7723, 7747, 7751, 7774, 7807, 7822, 7828, 7852, 7874, 7881, 7885, 7905,
            7913, 7949, 7965, 7966, 7985, 7990, 7993, 8036, 8051, 8075, 8092, 8095, 8114, 8117, 8152, 8160, 8172, 8180,
            8195, 8196, 8240, 8264, 8291, 8296, 8313, 8368, 8375, 8388, 8408, 8438, 8440, 8519, 8557, 8589, 8598, 8602,
            8652, 8658, 8724, 8760, 8764, 8786, 8803, 8814, 8827, 8855, 8857, 8867, 8919, 8923, 8924, 8933, 8959, 8968,
            9004, 9019, 9079, 9096, 9105, 9113, 9130, 9148, 9171, 9172, 9198, 9201, 9250, 9254, 9271, 9283, 9289, 9296,
            9322, 9324, 9325, 9348, 9400, 9404, 9418, 9427, 9428, 9440, 9469, 9487, 9497, 9512, 9517, 9519, 9530, 9558,
            9564, 9565, 9585, 9587, 9592, 9600, 9601, 9602, 9633, 9655, 9668, 9679, 9697, 9717, 9724, 9741, 9821, 9825,
            9826, 9829, 9864, 9867, 9869, 9890, 9930, 9939, 9954, 9968, 10020, 10021, 10026, 10060, 10112, 10119, 10126,
            10175, 10195, 10202, 10203, 10221, 10222, 10227, 10243, 10251, 10277, 10296, 10303, 10306, 10328, 10352,
            10361, 10370, 10394, 10408, 10439, 10456, 10464, 10466, 10471, 10479, 10504, 10509, 10510, 10810, 11081,
            11332, 11608, 11611, 11865, 12391, 12394, 12397, 12400, 12672, 12922, 12931, 13179, 13454, 13718, 14500,
            14518, 14776, 15298, 15557, 15835, 15840, 16127, 16128, 16387, 16634, 16645, 16652, 17167, 17170, 17959,
            17960, 17963, 18225, 21177, 21181, 21191, 21803, 21804, 21807, 22585, 22858, 22859, 22867
        ]

        root = Path(root)
        super(Chairs, self).__init__(root, n_frames, geometric_transform=geometric_transform,
                                     photometric_transform=photometric_transform)

    def collect_samples(self):

        samples = []
        for flow_map in sorted((self.root).glob('*.flo')):
            info = flow_map.splitall()
            filename = info[-1]
            fid = int(filename[0:5])

            # Include only train / valid indices (depending on the chosen split)
            if self.split == 'training':
                if fid in self.valid_indices:
                    continue
            elif self.split == 'valid':
                if fid not in self.valid_indices:
                    continue
            else:
                raise ValueError('Split {} is undefined'.format(self.split))

            s = {'imgs': [self.root / '{:05d}_img{:d}.ppm'.format(fid, i+1) for i in range(self.n_frames)]}
            try:
                assert all([p.is_file() for p in s['imgs']])

                if self.with_flow:
                    if self.n_frames == 2:
                        # for img1 img2, flow_12 will be evaluated
                        s['flow'] = flow_map
                        assert s['flow'].is_file()
                    else:
                        raise NotImplementedError(
                            'n_frames {} with flow or mask'.format(self.n_frames))

            except AssertionError:
                print('Incomplete sample for: {}'.format(s['imgs'][0]))
                continue
            samples.append(s)

        return samples


class KITTIFlowMV(ImgSeqDataset):
    """
    This dataset is used for unsupervised training only
    """

    def __init__(self, root, n_frames=2, geometric_transform=None, photometric_transform=None):
        super(KITTIFlowMV, self).__init__(root, n_frames, geometric_transform=geometric_transform,
                                        photometric_transform=photometric_transform)

    def collect_samples(self):
        img_dir = 'image_2'
        assert (self.root / img_dir).is_dir()

        samples = []
        for filename in sorted((self.root / img_dir).glob('*.png')):
            filename = filename.basename()
            root_filename = filename[:-7]

            img_list = (self.root / img_dir).files('*{}*.png'.format(root_filename))
            img_list.sort()

            for st in range(0, len(img_list) - self.n_frames + 1):
                seq = img_list[st:st + self.n_frames]
                sample = {}
                sample['imgs'] = []
                for i, file in enumerate(seq):
                    # We dont want to skip frames 10, 11
                    # frame_id = int(file[-6:-4])
                    #if 12 >= frame_id >= 9:
                    #    break
                    sample['imgs'].append(self.root.relpathto(file))
                if len(sample['imgs']) == self.n_frames:
                    samples.append(sample)

        return samples

class KITTIFlow(ImgSeqDataset):
    """
    This dataset is used for validation only, so all files about target are stored as
    file filepath and there is no transform about target.
    """

    def __init__(self, root, n_frames=2, with_flow=True, geometric_transform=None, photometric_transform=None):
        self.with_flow = with_flow

        super(KITTIFlow, self).__init__(root, n_frames, geometric_transform=geometric_transform,
                                        photometric_transform=photometric_transform)

    def collect_samples(self):
        '''Will search in training folder for folders 'flow_noc' or 'flow_occ'
               and 'colored_0' (KITTI 2012) or 'image_2' (KITTI 2015) '''
        flow_occ_dir = 'flow_occ'
        flow_noc_dir = 'flow_noc'
        img_dir = 'image_2' if (self.root / 'image_2').is_dir() else 'colored_0'
        assert (self.root / img_dir).is_dir()

        samples = []
        for flow_map in sorted((self.root / img_dir).glob('*_10.png')):
            flow_map = flow_map.basename()
            root_filename = flow_map[:-7]

            s = {}
            if self.with_flow:
                flow_occ_map = flow_occ_dir + '/' + flow_map
                flow_noc_map = flow_noc_dir + '/' + flow_map
                s.update({'flow_occ': flow_occ_map, 'flow_noc': flow_noc_map})

            img1 = img_dir + '/' + root_filename + '_10.png'
            img2 = img_dir + '/' + root_filename + '_11.png'
            assert (self.root / img1).is_file() and (self.root / img2).is_file()
            imgs = [img1, img2]
            if self.n_frames == 3:
                img0 = img_dir + '/' + root_filename + '_09.png'
                assert (self.root / img0).is_file()
                imgs = [img0,] + imgs

            s.update({'imgs': imgs})
            samples.append(s)
        return samples


class Things3D(ImgSeqDataset):
    def __init__(self, root, n_frames=2, split='training', with_flow=False, geometric_transform=None,
                 photometric_transform=None):
        self.with_flow = with_flow
        self.split = split

        root = Path(root)
        super(Things3D, self).__init__(root, n_frames, geometric_transform=geometric_transform,
                                      photometric_transform=photometric_transform)

    def collect_samples(self):
        if self.n_frames > 2:
            raise NotImplementedError('n_frames {}'.format(self.n_frames))
        if self.with_flow:
            raise NotImplementedError('with_flow {}'.format(self.with_flow))

        samples = []
        if self.split == 'training':
            path = self.root / Path('TRAIN')
        elif self.split == 'valid':
            path = self.root / Path('TEST')
        else:
            raise ValueError('Split {} is undefined'.format(self.split))

        for scene in sorted(path.glob('*/*')):
            images = [img for img in sorted(scene.glob('left/*.png'))]
            for i in range(len(images) - 1):
                s = {'imgs': [images[i], images[i + 1]]}
                assert all([p.is_file() for p in s['imgs']])
                samples.append(s)

        return samples

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from easydict import EasyDict
    from transforms.geometric_transforms import get_geometric_transforms
    from transforms.photometric_transforms import get_photometric_transforms
    from torch.utils.data import DataLoader

    geometric_transform = get_geometric_transforms(cfg=EasyDict({
        'crop': True,
        'crop_size': [100, 100],
        'hflip': True,
    }))
    photometric_transforms = get_photometric_transforms(cfg=EasyDict({
        'hue': 0.5,
        #'gamma': 0.5
        'swap_channels': True
    }))
    #dataset = Chairs(root="/home/deu/Datasets/FlyingChairs_release/data", with_flow=True,
    #                 geometric_transform=geometric_transform, photometric_transform=photometric_transforms)
    #dataset = KITTIFlow(root="/home/deu/Datasets/KITTI_2012/training", with_flow=True, geometric_transform=geometric_transform,
    #                    photometric_transform=photometric_transforms)
    #dataset = KITTIFlowMV(root="/home/deu/Datasets/KITTI_2015_multiview/training", n_frames=2, geometric_transform=None,
    #                    photometric_transform=None)
    dataset = Things3D(root="/home/deu/Datasets/FlyingThings3D/frames_cleanpass", n_frames=2, split='training',
                       geometric_transform=None, photometric_transform=photometric_transforms)
    loader = DataLoader(dataset, batch_size=1)
    print(len(dataset))
    for sample in loader:
        img1 = sample['img1_ph'][0].numpy().transpose(1, 2, 0)
        img2 = sample['img2_ph'][0].numpy().transpose(1, 2, 0)
        #img1_ph = sample['img1_ph'][0].numpy().transpose(1, 2, 0)
        #img2_ph = sample['img2_ph'][0].numpy().transpose(1, 2, 0)
        fig, ax = plt.subplots(2,1, figsize=(10,5))
        ax[0].imshow(img1)
        ax[1].imshow(img2)
        #ax[1,0].imshow(img1_ph)
        #ax[1,1].imshow(img2_ph)
        plt.show()