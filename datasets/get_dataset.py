import copy
from torchvision import transforms

from torch.utils.data import ConcatDataset
from transforms.co_transforms import get_co_transforms
from transforms.ar_transforms.ap_transforms import get_ap_transforms
from transforms import sep_transforms
from datasets.flow_datasets import SintelRaw, Sintel
from datasets.flow_datasets import KITTIRawFile, KITTIFlow, KITTIFlowMV
from datasets.flow_datasets import Chairs


def get_dataset(all_cfg):
    cfgs = all_cfg.data

    input_transform = transforms.Compose([
        sep_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
    ])

    co_transform = get_co_transforms(aug_args=all_cfg.data_aug)

    train_set = []
    valid_set = []

    # Collect all train / valid datasets into lists
    for cfg in cfgs:
        if cfg.type == 'Sintel_Flow':
            ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None

            train_set_1 = Sintel(cfg.root_sintel, n_frames=cfg.train_n_frames, type='clean',
                                 split=cfg.train_split, subsplit=cfg.train_subsplit,
                                 with_flow=False,
                                 ap_transform=ap_transform,
                                 transform=input_transform,
                                 co_transform=co_transform
                                 )
            train_set_2 = Sintel(cfg.root_sintel, n_frames=cfg.train_n_frames, type='final',
                                 split=cfg.train_split, subsplit=cfg.train_subsplit,
                                 with_flow=False,
                                 ap_transform=ap_transform,
                                 transform=input_transform,
                                 co_transform=co_transform
                                 )
            train_set += [train_set_1, train_set_2]

            valid_input_transform = copy.deepcopy(input_transform)
            valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))

            valid_set_1 = Sintel(cfg.root_sintel, n_frames=cfg.val_n_frames, type='clean',
                                 split=cfg.val_split, subsplit=cfg.val_subsplit,
                                 transform=valid_input_transform,
                                 target_transform={'flow': sep_transforms.ArrayToTensor()}
                                 )
            valid_set_2 = Sintel(cfg.root_sintel, n_frames=cfg.val_n_frames, type='final',
                                 split=cfg.val_split, subsplit=cfg.val_subsplit,
                                 transform=valid_input_transform,
                                 target_transform={'flow': sep_transforms.ArrayToTensor()}
                                 )
            valid_set += [valid_set_1, valid_set_2]

        elif cfg.type == 'Chairs':
            ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None

            train_set_1 = Chairs(cfg.root_chairs,  n_frames=cfg.train_n_frames, split='training', with_flow=False,
                               ap_transform=ap_transform, transform=input_transform, co_transform=co_transform)
                               #target_transform={'flow': sep_transforms.ArrayToTensor()})
            train_set += [train_set_1]

            valid_input_transform = copy.deepcopy(input_transform)
            valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))

            valid_set_1 = Chairs(cfg.root_chairs, n_frames=cfg.val_n_frames, split='valid', with_flow=True,
                               transform=valid_input_transform, target_transform={'flow': sep_transforms.ArrayToTensor()})
            valid_set += [valid_set_1]

        elif cfg.type == 'Sintel_Raw':
            train_set_1 = SintelRaw(cfg.root_sintel_raw, n_frames=cfg.train_n_frames,
                                  transform=input_transform, co_transform=co_transform)
            train_set += [train_set_1]

            valid_input_transform = copy.deepcopy(input_transform)
            valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))
            valid_set_1 = Sintel(cfg.root_sintel, n_frames=cfg.val_n_frames, type='clean',
                                 split='training', subsplit=cfg.val_subsplit,
                                 transform=valid_input_transform,
                                 target_transform={'flow': sep_transforms.ArrayToTensor()}
                                 )
            valid_set_2 = Sintel(cfg.root_sintel, n_frames=cfg.val_n_frames, type='final',
                                 split='training', subsplit=cfg.val_subsplit,
                                 transform=valid_input_transform,
                                 target_transform={'flow': sep_transforms.ArrayToTensor()}
                                 )
            valid_set += [valid_set_1, valid_set_2]
        elif cfg.type == 'KITTI_Raw':
            train_input_transform = copy.deepcopy(input_transform)
            train_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape))

            ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None
            train_set_1 = KITTIRawFile(
                cfg.root,
                cfg.train_file,
                cfg.train_n_frames,
                transform=train_input_transform,
                ap_transform=ap_transform,
                co_transform=co_transform  # no target here
            )
            train_set += [train_set_1]

            valid_input_transform = copy.deepcopy(input_transform)
            valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))

            valid_set_1 = KITTIFlow(cfg.root_kitti15, n_frames=cfg.val_n_frames,
                                    transform=valid_input_transform,
                                    )
            valid_set_2 = KITTIFlow(cfg.root_kitti12, n_frames=cfg.val_n_frames,
                                    transform=valid_input_transform,
                                    )
            valid_set += [valid_set_1, valid_set_2]
        elif cfg.type == 'KITTI_MV':
            train_input_transform = copy.deepcopy(input_transform)
            train_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.train_shape))

            root_flow = cfg.root_kitti15 if cfg.train_15 else cfg.root_kitti12

            ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None
            train_set_1 = KITTIFlowMV(
                root_flow,
                cfg.train_n_frames,
                transform=train_input_transform,
                ap_transform=ap_transform,
                co_transform=co_transform  # no target here
            )
            train_set += [train_set_1]

            valid_input_transform = copy.deepcopy(input_transform)
            valid_input_transform.transforms.insert(0, sep_transforms.Zoom(*cfg.test_shape))

            valid_set_1 = KITTIFlow(cfg.root_kitti15, n_frames=cfg.val_n_frames,
                                    transform=valid_input_transform,
                                    )
            valid_set_2 = KITTIFlow(cfg.root_kitti12, n_frames=cfg.val_n_frames,
                                    transform=valid_input_transform,
                                    )
            valid_set = [valid_set_1, valid_set_2]
        else:
            raise NotImplementedError(cfg.type)

    # Concatenate only train sets
    return ConcatDataset(train_set), valid_set