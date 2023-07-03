import imageio
import numpy as np
import random
from path import Path
from abc import abstractmethod, ABCMeta
from torch.utils.data import Dataset
from utils.flow_utils import load_flow


class ImgSeqDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, root, n_frames, input_transform=None, co_transform=None,
                 target_transform=None, ap_transform=None):
        self.root = Path(root)
        self.n_frames = n_frames
        self.input_transform = input_transform
        self.co_transform = co_transform
        self.ap_transform = ap_transform
        self.target_transform = target_transform
        self.samples = self.collect_samples()

    @abstractmethod
    def collect_samples(self):
        pass

    def _load_sample(self, s):
        images = s['imgs']
        images = [imageio.imread(self.root / p).astype(np.float32) for p in images]

        target = {}
        if 'flow' in s:
            target['flow'] = load_flow(self.root / s['flow'])
        if 'mask' in s:
            # 0~255 HxWx1
            mask = imageio.imread(self.root / s['mask']).astype(np.float32) / 255.
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            target['mask'] = np.expand_dims(mask, -1)
        return images, target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        images, target = self._load_sample(self.samples[idx])

        if self.co_transform is not None:
            # In unsupervised learning, there is no need to change target with image
            images, _ = self.co_transform(images, {})
        if self.input_transform is not None:
            images = [self.input_transform(i) for i in images]
        data = {'img{}'.format(i + 1): p for i, p in enumerate(images)}

        if self.ap_transform is not None:
            imgs_ph = self.ap_transform(
                [data['img{}'.format(i + 1)].clone() for i in range(self.n_frames)])
            for i in range(self.n_frames):
                data['img{}_ph'.format(i + 1)] = imgs_ph[i]

        if self.target_transform is not None:
            for key in self.target_transform.keys():
                target[key] = self.target_transform[key](target[key])
        data['target'] = target
        return data


class SintelRaw(ImgSeqDataset):
    def __init__(self, root, n_frames=2, transform=None, co_transform=None):
        super(SintelRaw, self).__init__(root, n_frames, input_transform=transform,
                                        co_transform=co_transform)

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
    def __init__(self, root, n_frames=2, type='clean', split='training',
                 subsplit='trainval', with_flow=True, ap_transform=None,
                 transform=None, target_transform=None, co_transform=None, ):
        self.dataset_type = type
        self.with_flow = with_flow

        self.split = split
        self.subsplit = subsplit
        self.training_scene = ['alley_1', 'ambush_4', 'ambush_6', 'ambush_7', 'bamboo_2',
                               'bandage_2', 'cave_2', 'market_2', 'market_5', 'shaman_2',
                               'sleeping_2', 'temple_3']  # Unofficial train-val split

        root = Path(root) / split
        super(Sintel, self).__init__(root, n_frames, input_transform=transform,
                                     target_transform=target_transform,
                                     co_transform=co_transform, ap_transform=ap_transform)

    def collect_samples(self):
        img_dir = self.root / Path(self.dataset_type)
        flow_dir = self.root / 'flow'

        assert img_dir.isdir() and flow_dir.isdir()

        samples = []
        for flow_map in sorted((self.root / flow_dir).glob('*/*.flo')):
            info = flow_map.splitall()
            scene, filename = info[-2:]
            fid = int(filename[-8:-4])
            if self.split == 'training' and self.subsplit != 'trainval':
                if self.subsplit == 'train' and scene not in self.training_scene:
                    continue
                if self.subsplit == 'val' and scene in self.training_scene:
                    continue

            s = {'imgs': [img_dir / scene / 'frame_{:04d}.png'.format(fid + i) for i in
                          range(self.n_frames)]}
            try:
                assert all([p.isfile() for p in s['imgs']])

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
                        assert s['flow'].isfile()
            except AssertionError:
                print('Incomplete sample for: {}'.format(s['imgs'][0]))
                continue
            samples.append(s)

        return samples


class Chairs(ImgSeqDataset):
    def __init__(self, root, n_frames=2, split='training', with_flow=True, ap_transform=None, transform=None,
                 target_transform=None, co_transform=None):
        self.with_flow = with_flow
        self.split = split
        self.valid_indices = [
            5, 17, 42, 45, 58, 62, 96, 111, 117, 120, 121, 131, 132,
            152, 160, 248, 263, 264, 291, 293, 295, 299, 316, 320, 336,
            337, 343, 358, 399, 401, 429, 438, 468, 476, 494, 509, 528,
            531, 572, 581, 583, 588, 593, 681, 688, 696, 714, 767, 786,
            810, 825, 836, 841, 883, 917, 937, 942, 970, 974, 980, 1016,
            1043, 1064, 1118, 1121, 1133, 1153, 1155, 1158, 1159, 1173,
            1187, 1219, 1237, 1238, 1259, 1266, 1278, 1296, 1354, 1378,
            1387, 1494, 1508, 1518, 1574, 1601, 1614, 1668, 1673, 1699,
            1712, 1714, 1737, 1841, 1872, 1879, 1901, 1921, 1934, 1961,
            1967, 1978, 2018, 2030, 2039, 2043, 2061, 2113, 2204, 2216,
            2236, 2250, 2274, 2292, 2310, 2342, 2359, 2374, 2382, 2399,
            2415, 2419, 2483, 2502, 2504, 2576, 2589, 2590, 2622, 2624,
            2636, 2651, 2655, 2658, 2659, 2664, 2672, 2706, 2707, 2709,
            2725, 2732, 2761, 2827, 2864, 2866, 2905, 2922, 2929, 2966,
            2972, 2993, 3010, 3025, 3031, 3040, 3041, 3070, 3113, 3124,
            3129, 3137, 3141, 3157, 3183, 3206, 3219, 3247, 3253, 3272,
            3276, 3321, 3328, 3333, 3338, 3341, 3346, 3351, 3396, 3419,
            3430, 3433, 3448, 3455, 3463, 3503, 3526, 3529, 3537, 3555,
            3577, 3584, 3591, 3594, 3597, 3603, 3613, 3615, 3670, 3676,
            3678, 3697, 3723, 3728, 3734, 3745, 3750, 3752, 3779, 3782,
            3813, 3817, 3819, 3854, 3885, 3944, 3947, 3970, 3985, 4011,
            4022, 4071, 4075, 4132, 4158, 4167, 4190, 4194, 4207, 4246,
            4249, 4298, 4307, 4317, 4318, 4319, 4320, 4382, 4399, 4401,
            4407, 4416, 4423, 4484, 4491, 4493, 4517, 4525, 4538, 4578,
            4606, 4609, 4620, 4623, 4637, 4646, 4662, 4668, 4716, 4739,
            4747, 4770, 4774, 4776, 4785, 4800, 4845, 4863, 4891, 4904,
            4922, 4925, 4956, 4963, 4964, 4994, 5011, 5019, 5036, 5038,
            5041, 5055, 5118, 5122, 5130, 5162, 5164, 5178, 5196, 5227,
            5266, 5270, 5273, 5279, 5299, 5310, 5314, 5363, 5375, 5384,
            5393, 5414, 5417, 5433, 5448, 5494, 5505, 5509, 5525, 5566,
            5581, 5602, 5609, 5620, 5653, 5670, 5678, 5690, 5700, 5703,
            5724, 5752, 5765, 5803, 5811, 5860, 5881, 5895, 5912, 5915,
            5940, 5952, 5966, 5977, 5988, 6007, 6037, 6061, 6069, 6080,
            6111, 6127, 6146, 6161, 6166, 6168, 6178, 6182, 6190, 6220,
            6235, 6253, 6270, 6343, 6372, 6379, 6410, 6411, 6442, 6453,
            6481, 6498, 6500, 6509, 6532, 6541, 6543, 6560, 6576, 6580,
            6594, 6595, 6609, 6625, 6629, 6644, 6658, 6673, 6680, 6698,
            6699, 6702, 6705, 6741, 6759, 6785, 6792, 6794, 6809, 6810,
            6830, 6838, 6869, 6871, 6889, 6925, 6995, 7003, 7026, 7029,
            7080, 7082, 7097, 7102, 7116, 7165, 7200, 7232, 7271, 7282,
            7324, 7333, 7335, 7372, 7387, 7407, 7472, 7474, 7482, 7489,
            7499, 7516, 7533, 7536, 7566, 7620, 7654, 7691, 7704, 7722,
            7746, 7750, 7773, 7806, 7821, 7827, 7851, 7873, 7880, 7884,
            7904, 7912, 7948, 7964, 7965, 7984, 7989, 7992, 8035, 8050,
            8074, 8091, 8094, 8113, 8116, 8151, 8159, 8171, 8179, 8194,
            8195, 8239, 8263, 8290, 8295, 8312, 8367, 8374, 8387, 8407,
            8437, 8439, 8518, 8556, 8588, 8597, 8601, 8651, 8657, 8723,
            8759, 8763, 8785, 8802, 8813, 8826, 8854, 8856, 8866, 8918,
            8922, 8923, 8932, 8958, 8967, 9003, 9018, 9078, 9095, 9104,
            9112, 9129, 9147, 9170, 9171, 9197, 9200, 9249, 9253, 9270,
            9282, 9288, 9295, 9321, 9323, 9324, 9347, 9399, 9403, 9417,
            9426, 9427, 9439, 9468, 9486, 9496, 9511, 9516, 9518, 9529,
            9557, 9563, 9564, 9584, 9586, 9591, 9599, 9600, 9601, 9632,
            9654, 9667, 9678, 9696, 9716, 9723, 9740, 9820, 9824, 9825,
            9828, 9863, 9866, 9868, 9889, 9929, 9938, 9953, 9967, 10019,
            10020, 10025, 10059, 10111, 10118, 10125, 10174, 10194,
            10201, 10202, 10220, 10221, 10226, 10242, 10250, 10276,
            10295, 10302, 10305, 10327, 10351, 10360, 10369, 10393,
            10407, 10438, 10455, 10463, 10465, 10470, 10478, 10503,
            10508, 10509, 10809, 11080, 11331, 11607, 11610, 11864,
            12390, 12393, 12396, 12399, 12671, 12921, 12930, 13178,
            13453, 13717, 14499, 14517, 14775, 15297, 15556, 15834,
            15839, 16126, 16127, 16386, 16633, 16644, 16651, 17166,
            17169, 17958, 17959, 17962, 18224, 21176, 21180, 21190,
            21802, 21803, 21806, 22584, 22857, 22858, 22866
        ]

        root = Path(root)
        super(Chairs, self).__init__(root, n_frames, input_transform=transform,
                                     target_transform=target_transform,
                                     co_transform=co_transform, ap_transform=ap_transform)

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
            else:
                if fid not in self.valid_indices:
                    continue

            s = {'imgs': [self.root / '{:05d}_img{:d}.ppm'.format(fid, i+1) for i in range(self.n_frames)]}
            try:
                assert all([p.isfile() for p in s['imgs']])

                if self.with_flow:
                    if self.n_frames == 2:
                        # for img1 img2, flow_12 will be evaluated
                        s['flow'] = flow_map
                        assert s['flow'].isfile()
                    else:
                        raise NotImplementedError(
                            'n_frames {} with flow or mask'.format(self.n_frames))

            except AssertionError:
                print('Incomplete sample for: {}'.format(s['imgs'][0]))
                continue
            samples.append(s)

        return samples


class KITTIRawFile(ImgSeqDataset):
    def __init__(self, root, sp_file, n_frames=2, ap_transform=None,
                 transform=None, target_transform=None, co_transform=None):
        self.sp_file = sp_file
        super(KITTIRawFile, self).__init__(root, n_frames,
                                           input_transform=transform,
                                           target_transform=target_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform)

    def collect_samples(self):
        samples = []
        with open(self.sp_file, 'r') as f:
            for line in f.readlines():
                sp = line.split()
                s = {'imgs': [sp[i] for i in range(self.n_frames)]}
                samples.append(s)
            return samples


class KITTIFlowMV(ImgSeqDataset):
    """
    This dataset is used for unsupervised training only
    """

    def __init__(self, root, n_frames=2,
                 transform=None, co_transform=None, ap_transform=None, ):
        super(KITTIFlowMV, self).__init__(root, n_frames,
                                          input_transform=transform,
                                          co_transform=co_transform,
                                          ap_transform=ap_transform)

    def collect_samples(self):
        flow_occ_dir = 'flow_' + 'occ'
        assert (self.root / flow_occ_dir).isdir()

        img_l_dir, img_r_dir = 'image_2', 'image_3'
        assert (self.root / img_l_dir).isdir() and (self.root / img_r_dir).isdir()

        samples = []
        for flow_map in sorted((self.root / flow_occ_dir).glob('*.png')):
            flow_map = flow_map.basename()
            root_filename = flow_map[:-7]

            for img_dir in [img_l_dir, img_r_dir]:
                img_list = (self.root / img_dir).files('*{}*.png'.format(root_filename))
                img_list.sort()

                for st in range(0, len(img_list) - self.n_frames + 1):
                    seq = img_list[st:st + self.n_frames]
                    sample = {}
                    sample['imgs'] = []
                    for i, file in enumerate(seq):
                        frame_id = int(file[-6:-4])
                        if 12 >= frame_id >= 9:
                            break
                        sample['imgs'].append(self.root.relpathto(file))
                    if len(sample['imgs']) == self.n_frames:
                        samples.append(sample)
        return samples


class KITTIFlow(ImgSeqDataset):
    """
    This dataset is used for validation only, so all files about target are stored as
    file filepath and there is no transform about target.
    """

    def __init__(self, root, n_frames=2, transform=None):
        super(KITTIFlow, self).__init__(root, n_frames, input_transform=transform)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # img 1 2 for 2 frames, img 0 1 2 for 3 frames.
        st = 1 if self.n_frames == 2 else 0
        ed = st + self.n_frames
        imgs = [s['img{}'.format(i)] for i in range(st, ed)]

        inputs = [imageio.imread(self.root / p).astype(np.float32) for p in imgs]
        raw_size = inputs[0].shape[:2]

        data = {
            'flow_occ': self.root / s['flow_occ'],
            'flow_noc': self.root / s['flow_noc'],
        }

        data.update({  # for test set
            'im_shape': raw_size,
            'img1_path': self.root / s['img1'],
        })

        if self.input_transform is not None:
            inputs = [self.input_transform(i) for i in inputs]
        data.update({'img{}'.format(i + 1): inputs[i] for i in range(self.n_frames)})
        return data

    def collect_samples(self):
        '''Will search in training folder for folders 'flow_noc' or 'flow_occ'
               and 'colored_0' (KITTI 2012) or 'image_2' (KITTI 2015) '''
        flow_occ_dir = 'flow_' + 'occ'
        flow_noc_dir = 'flow_' + 'noc'
        assert (self.root / flow_occ_dir).isdir()

        img_dir = 'image_2'
        assert (self.root / img_dir).isdir()

        samples = []
        for flow_map in sorted((self.root / flow_occ_dir).glob('*.png')):
            flow_map = flow_map.basename()
            root_filename = flow_map[:-7]

            flow_occ_map = flow_occ_dir + '/' + flow_map
            flow_noc_map = flow_noc_dir + '/' + flow_map
            s = {'flow_occ': flow_occ_map, 'flow_noc': flow_noc_map}

            img1 = img_dir + '/' + root_filename + '_10.png'
            img2 = img_dir + '/' + root_filename + '_11.png'
            assert (self.root / img1).isfile() and (self.root / img2).isfile()
            s.update({'img1': img1, 'img2': img2})
            if self.n_frames == 3:
                img0 = img_dir + '/' + root_filename + '_09.png'
                assert (self.root / img0).isfile()
                s.update({'img0': img0})
            samples.append(s)
        return samples


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = Chairs(root="/home/deu/FlyingChairs_release/data")
    sample = dataset.__getitem__(0)
    img1 = sample['img1'] / 255.0
    img2 = sample['img2'] / 255.0
    fig, ax = plt.subplots(2)
    ax[0].imshow(img1)
    ax[1].imshow(img2)
    plt.show()