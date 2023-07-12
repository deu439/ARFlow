from .flow_loss import unFlowLoss
from .elbo_loss import ElboLoss

def get_loss(cfg):
    if cfg.type == 'unflow':
        loss = unFlowLoss(cfg)
    elif cfg.type == 'elbo':
        loss = ElboLoss(cfg)
    else:
        raise NotImplementedError(cfg.type)
    return loss
