from .pwclite import PWCLite
from .pwclite_prob import PWCLiteProb
from .pwclite_uflow import PWCLiteUflow
from .uflow_model import PWCFlow


def get_model(cfg):
    if cfg.type == 'pwclite':
        model = PWCLite(cfg)
    elif cfg.type == 'pwclite_prob':
        model = PWCLiteProb(cfg)
    elif cfg.type == 'pwclite_uflow':
        model = PWCLiteUflow(cfg)
    elif cfg.type == 'uflow':
        model = PWCFlow(cfg)
    else:
        raise NotImplementedError(cfg.type)
    return model
