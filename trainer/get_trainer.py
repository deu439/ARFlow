from . import uflow_trainer
from . import uflow_elbo_trainer

def get_trainer(name):
    if name == 'uflow':
        TrainFramework = uflow_trainer.TrainFramework
    elif name == 'uflow_elbo':
        TrainFramework = uflow_elbo_trainer.TrainFramework
    else:
        raise NotImplementedError(name)

    return TrainFramework
