import torch
from utils.torch_utils import init_seed

from datasets.get_dataset import get_dataset
from models.get_model import get_model
from losses.get_loss import get_loss
from trainer.get_trainer import get_trainer


def main(cfg, _log):
    init_seed(cfg.seed)

    _log.info("=> fetching img pairs.")
    train_set, valid_set = get_dataset(cfg)

    valid_len = sum([len(s) for s in valid_set])
    _log.info('{} samples found, {} train samples and {} test samples '.format(
        valid_len + len(train_set),
        len(train_set),
        valid_len))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.train.batch_size,
        num_workers=cfg.train.workers, pin_memory=True, shuffle=True)

    # Default validation batch size is 1 for compatibility with KITTI dataset
    valid_batch_size = cfg.train.valid_batch_size if hasattr(cfg.train, 'valid_batch_size') else 1
    valid_loader = [torch.utils.data.DataLoader(
        s, batch_size=valid_batch_size,
        num_workers=min(4, cfg.train.workers),
        pin_memory=True, shuffle=True) for s in valid_set]
    valid_size = sum([len(l) for l in valid_loader])

    if cfg.train.epoch_size == 0:
        cfg.train.epoch_size = len(train_loader)
    if cfg.train.valid_size == 0:
        cfg.train.valid_size = valid_size
    cfg.train.epoch_size = min(cfg.train.epoch_size, len(train_loader))
    cfg.train.valid_size = min(cfg.train.valid_size, valid_size)

    model = get_model(cfg.model)
    loss = get_loss(cfg.loss)
    trainer = get_trainer(cfg.trainer)(
        train_loader, valid_loader, model, loss, _log, cfg.save_root, cfg.train)

    trainer.train()
