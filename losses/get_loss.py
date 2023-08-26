from .flow_loss import unFlowLoss
from .elbo_loss import ElboLoss
from .fullres_loss import FullResLoss
from .uflow_loss import UFlowLoss

def get_loss(cfg):
    if cfg.type == 'unflow':
        loss = unFlowLoss(cfg)
    elif cfg.type == 'elbo':
        loss = ElboLoss(cfg)
    elif cfg.type == 'fullres':
        loss = FullResLoss(cfg)
    elif cfg.type == 'uflow':
        loss = UFlowLoss(cfg)
    else:
        raise NotImplementedError(cfg.type)
    return loss
