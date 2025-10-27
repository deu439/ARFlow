from torch.utils.data import ConcatDataset
from transforms.geometric_transforms import get_geometric_transforms, Compose, Scale
from transforms.photometric_transforms import get_photometric_transforms
from datasets.flow_datasets import Sintel
from datasets.flow_datasets import KITTIFlow, KITTIFlowMV
from datasets.flow_datasets import Chairs, Chairs2
from datasets.flow_datasets import Things3D


def get_dataset(all_cfg):
    cfgs = all_cfg.data


    train_set = []
    valid_set = []

    # Collect all train / valid datasets into lists
    for cfg in cfgs:
        # Configure training augmentations
        geometric_transform = get_geometric_transforms(cfg=cfg.geometric_aug) \
            if hasattr(cfg, 'geometric_aug') else None
        photometric_transform = get_photometric_transforms(cfg=cfg.photometric_aug) \
            if hasattr(cfg, 'photometric_aug') else None

        # Configure validation scaling
        valid_transform = Compose([Scale(size=cfg.test_shape), ]) if hasattr(cfg, 'test_shape') else None

        if cfg.name == 'Sintel':
            if cfg.type == 'train':
                train_set_1 = Sintel(cfg.root_sintel, n_frames=cfg.n_frames, split=cfg.split,
                                     type='clean' if cfg.clean else 'final',
                                     subsplit=cfg.subsplit, with_flow=False, geometric_transform=geometric_transform,
                                     photometric_transform=photometric_transform)
                train_set += [train_set_1]

            else:
                valid_set_1 = Sintel(cfg.root_sintel, n_frames=cfg.n_frames, split=cfg.split,
                                     type='clean' if cfg.clean else 'final',
                                     subsplit=cfg.subsplit, with_flow=cfg.get('with_flow', True),
                                     geometric_transform=valid_transform)
                valid_set += [valid_set_1]

        elif cfg.name == 'Chairs2':
            if cfg.type == 'train':
                train_set_1 = Chairs2(cfg.root_chairs, n_frames=cfg.n_frames, split=cfg.split, with_flow=False,
                                      geometric_transform=geometric_transform,
                                      photometric_transform=photometric_transform)

                train_set += [train_set_1]
            else:
                valid_set_1 = Chairs2(cfg.root_chairs, n_frames=cfg.n_frames, split=cfg.split,
                                      with_flow=cfg.get('with_flow', True), geometric_transform=valid_transform)
                valid_set += [valid_set_1]

        elif cfg.name == 'Chairs':
            if cfg.type == 'train':
                train_set_1 = Chairs(cfg.root_chairs, n_frames=cfg.n_frames, split=cfg.split, with_flow=False,
                                     geometric_transform=geometric_transform,
                                     photometric_transform=photometric_transform)
                train_set += [train_set_1]
            else:
                valid_set_1 = Chairs(cfg.root_chairs, n_frames=cfg.n_frames, split=cfg.split,
                                     with_flow=cfg.get('with_flow', True), geometric_transform=valid_transform)
                valid_set += [valid_set_1]

        elif cfg.name == 'KITTI':
            if cfg.type == 'train':
                train_set_1 = KITTIFlow(cfg.root, n_frames=cfg.n_frames, split=cfg.split, with_flow=False,
                                           geometric_transform=geometric_transform,
                                           photometric_transform=photometric_transform)
                train_set += [train_set_1]

            else:
                valid_set_1 = KITTIFlow(cfg.root, n_frames=cfg.n_frames, split=cfg.split,
                                        geometric_transform=valid_transform, with_flow=cfg.get('with_flow', True))
                valid_set += [valid_set_1]

        elif cfg.name == 'KITTIMV':
            if cfg.type == 'train':
                train_set_1 = KITTIFlowMV(cfg.root, n_frames=cfg.n_frames,
                                        geometric_transform=geometric_transform,
                                        photometric_transform=photometric_transform)
                train_set += [train_set_1]

            else:
                valid_set_1 = KITTIFlowMV(cfg.root, n_frames=cfg.n_frames, geometric_transform=valid_transform)
                valid_set += [valid_set_1]

        elif cfg.name == 'Things':
            if cfg.type == 'train':
                train_set_1 = Things3D(cfg.root, n_frames=cfg.n_frames, split=cfg.split,
                                       geometric_transform=geometric_transform,
                                       photometric_transform=photometric_transform)
                train_set += [train_set_1]
            else:
                raise NotImplementedError(cfg.type)
        else:
            raise NotImplementedError(cfg.name)

    # Concatenate only train sets
    return ConcatDataset(train_set) if len(train_set) > 0 else None, valid_set