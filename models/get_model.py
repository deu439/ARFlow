from .pwclite import PWCLite
from .pwclite_prob import PWCLiteProb


def get_model(cfg):
    if cfg.type == 'pwclite':
        model = PWCLite(cfg)
    elif cfg.type == 'pwclite_prob':
        model = PWCLiteProb(cfg)
    else:
        raise NotImplementedError(cfg.type)
    return model
