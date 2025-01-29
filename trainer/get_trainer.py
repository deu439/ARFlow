from . import sintel_trainer
from . import kitti_trainer
from . import chairs_trainer
from . import chairs_elbo_trainer
from . import sintel_elbo_trainer

def get_trainer(name):
    if name == 'Sintel':
        TrainFramework = sintel_trainer.TrainFramework
    elif name == 'KITTI':
        TrainFramework = kitti_trainer.TrainFramework
    elif name == 'Chairs':
        TrainFramework = chairs_trainer.TrainFramework
    elif name == 'ChairsElbo':
        TrainFramework = chairs_elbo_trainer.TrainFramework
    elif name == 'SintelElbo':
        TrainFramework = sintel_elbo_trainer.TrainFramework
    elif name == 'ChairsMse':
        TrainFramework = chairs_mse_trainer.TrainFramework
    else:
        raise NotImplementedError(name)

    return TrainFramework
