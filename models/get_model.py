from .pwclite import PWCLite
from .pwclite_prob import PWCLiteProb
from .pwclite_uflow import PWCLiteUflow
from .uflow_model import PWCFlow
from .uflow_prob_model import PWCProbFlow, ComponentNet


def get_model(cfg):
    if cfg.type == 'pwclite':
        model = PWCLite(cfg)
    elif cfg.type == 'pwclite_prob':
        model = PWCLiteProb(cfg)
    elif cfg.type == 'pwclite_uflow':
        model = PWCLiteUflow(cfg)
    elif cfg.type == 'uflow':
        model = PWCFlow(cfg)
    elif cfg.type =='uflow_prob':
        model = PWCProbFlow(cfg)
    elif cfg.type == 'flownet_prob':
        model = FlowNetProbOut(cfg)
    elif cfg.type == 'component':
        model = ComponentNet(cfg)
    else:
        raise NotImplementedError(cfg.type)
    return model
