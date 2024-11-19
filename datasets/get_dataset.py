from torch.utils.data import ConcatDataset
from transforms.geometric_transforms import get_geometric_transforms
from transforms.photometric_transforms import get_photometric_transforms
from datasets.flow_datasets import SintelRaw, Sintel
from datasets.flow_datasets import KITTIRawFile, KITTIFlow, KITTIFlowMV
from datasets.flow_datasets import Chairs, Chairs2


def get_dataset(all_cfg):
    cfgs = all_cfg.data


    train_set = []
    valid_set = []

    # Collect all train / valid datasets into lists
    for cfg in cfgs:
        geometric_transform = get_geometric_transforms(cfg=cfg.geometric_aug) \
            if hasattr(cfg, 'geometric_aug') else None
        photometric_transform = get_photometric_transforms(cfg=cfg.photometric_aug) \
            if hasattr(cfg, 'photometric_aug') else None

        if cfg.type == 'Sintel_Flow':
            if cfg.split == 'train':
                ph_transform = get_photometric_transforms(cfg.ph_augmentation) if hasattr(cfg, 'at_cfg') else None
                train_set_1 = Sintel(cfg.root_sintel, n_frames=cfg.train_n_frames, type='clean', subsplit=cfg.subsplit,
                                     with_flow=False, geometric_transform=geometric_transform,
                                     photometric_transform=photometric_transform)
                train_set_2 = Sintel(cfg.root_sintel, n_frames=cfg.train_n_frames, type='final', subsplit=cfg.subsplit,
                                     with_flow=False, geometric_transform=geometric_transform,
                                     photometric_transform=photometric_transform)

                train_set += [train_set_1, train_set_2]
            else:
                valid_set_1 = Sintel(cfg.root_sintel, n_frames=cfg.n_frames, type='clean', subsplit=cfg.subsplit,
                                     with_flow=True)
                valid_set_2 = Sintel(cfg.root_sintel, n_frames=cfg.n_frames, type='final', subsplit=cfg.subsplit,
                                     with_flow=True)
                valid_set += [valid_set_1, valid_set_2]

        elif cfg.type == 'Chairs2':
            if cfg.split == 'train':
                train_set_1 = Chairs2(cfg.root_chairs, n_frames=cfg.n_frames, split='training', with_flow=False,
                                      geometric_transform=geometric_transform,
                                      photometric_transform=photometric_transform)

                train_set += [train_set_1]
            else:
                valid_set_1 = Chairs2(cfg.root_chairs, n_frames=cfg.n_frames, split='valid', with_flow=True)
                valid_set += [valid_set_1]

        elif cfg.type == 'Chairs':
            if cfg.split == 'train':
                train_set_1 = Chairs(cfg.root_chairs, n_frames=cfg.n_frames, split='training', with_flow=False,
                                     geometric_transform=geometric_transform,
                                     photometric_transform=photometric_transform)
                train_set += [train_set_1]
            else:
                valid_set_1 = Chairs(cfg.root_chairs, n_frames=cfg.n_frames, split='valid', with_flow=True)
                valid_set += [valid_set_1]

        elif cfg.type == 'Sintel_Raw':
            train_set_1 = SintelRaw(cfg.root_sintel_raw, n_frames=cfg.train_n_frames,
                                    geometric_transform=geometric_transform,
                                    photometric_transform=photometric_transform)
            train_set += [train_set_1]

            valid_set_1 = Sintel(cfg.root_sintel, n_frames=cfg.val_n_frames, type='clean', subsplit=cfg.val_subsplit)
            valid_set_2 = Sintel(cfg.root_sintel, n_frames=cfg.val_n_frames, type='final', subsplit=cfg.val_subsplit)
            valid_set += [valid_set_1, valid_set_2]
        elif cfg.type == 'KITTI_Raw':
            train_set_1 = KITTIRawFile(cfg.root, cfg.train_file, cfg.train_n_frames,
                                       geometric_transform=geometric_transform,
                                       photometric_transform=photometric_transform)
            train_set += [train_set_1]

            valid_set_1 = KITTIFlow(cfg.root_kitti15, n_frames=cfg.val_n_frames)
            valid_set_2 = KITTIFlow(cfg.root_kitti12, n_frames=cfg.val_n_frames)
            valid_set += [valid_set_1, valid_set_2]
        elif cfg.type == 'KITTI_MV':
            root_flow = cfg.root_kitti15 if cfg.train_15 else cfg.root_kitti12
            train_set_1 = KITTIFlowMV(root_flow, cfg.train_n_frames, geometric_transform=geometric_transform,
                                      photometric_transform=photometric_transform)
            train_set += [train_set_1]

            valid_set_1 = KITTIFlow(cfg.root_kitti15, n_frames=cfg.val_n_frames,
                                    geometric_transform=geometric_transform,
                                    photometric_transform=photometric_transform)
            valid_set_2 = KITTIFlow(cfg.root_kitti12, n_frames=cfg.val_n_frames,
                                    geometric_transform=geometric_transform,
                                    photometric_transform=photometric_transform)
            valid_set = [valid_set_1, valid_set_2]
        else:
            raise NotImplementedError(cfg.type)

    # Concatenate only train sets
    return ConcatDataset(train_set), valid_set