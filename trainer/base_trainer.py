import torch
import numpy as np
from abc import abstractmethod
from tensorboardX import SummaryWriter
from utils.torch_utils import bias_parameters, weight_parameters, \
    load_checkpoint, save_checkpoint, AdamW


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, train_loader, valid_loader, model, loss_func,
                 _log, save_root, config):
        self._log = _log

        self.cfg = config
        self.save_root = save_root
        self.summary_writer = SummaryWriter(str(save_root))

        self.train_loader, self.valid_loader = train_loader, valid_loader
        self.device, self.device_ids = self._prepare_device(config['n_gpu'])

        self.model = self._init_model(model)
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = self._create_lr_scheduler()
        self.loss_func = loss_func

        self.best_error = np.inf
        self.i_epoch = 0
        self.i_iter = 0

    @abstractmethod
    def _run_one_epoch(self):
        ...

    @abstractmethod
    def _validate_with_gt(self):
        ...

    def train(self):
        for epoch in range(self.cfg.epoch_num):
            self._run_one_epoch()

            if self.i_epoch % self.cfg.val_epoch_size == 0:
                errors, error_names = self._validate_with_gt()
                valid_res = ' '.join(
                    '{}: {:.2f}'.format(*t) for t in zip(error_names, errors))
                self._log.info(' * Epoch {} '.format(self.i_epoch) + valid_res)

            # Start learning rate decay after defined number of epochs
            if self.i_epoch >= self.cfg.lr_decay_start_epoch:
                self.lr_scheduler.step()
                self._log.info(' * lr: {}'.format(self.lr_scheduler.get_last_lr()))

    def _init_model(self, model):
        model = model.to(self.device)
        if self.cfg.pretrained_model:
            self._log.info("=> using pre-trained weights {}.".format(
                self.cfg.pretrained_model))
            epoch, weights = load_checkpoint(self.cfg.pretrained_model, self.device)

            from collections import OrderedDict
            new_weights = OrderedDict()
            model_keys = list(model.state_dict().keys())
            weight_keys = list(weights.keys())
            for a, b in zip(model_keys, weight_keys):
                new_weights[a] = weights[b]
            weights = new_weights
            model.load_state_dict(weights)
        else:
            self._log.info("=> Train from scratch.")
            model.init_weights()
        model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        return model

    def _create_optimizer(self):
        self._log.info('=> setting {} solver'.format(self.cfg.optim))

        # Split parameters into with and without decay, credit: https://github.com/karpathy/minGPT.git
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d)
        blacklist_weight_modules = (torch.nn.BatchNorm2d, )
        for mn, m in self.model.module.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.model.module.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        param_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.cfg.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": self.cfg.bias_decay},
        ]

        if self.cfg.optim == 'adamw':
            optimizer = AdamW(param_groups, self.cfg.lr,
                              betas=(self.cfg.momentum, self.cfg.beta))
        elif self.cfg.optim == 'adam':
            optimizer = torch.optim.Adam(param_groups, self.cfg.lr,
                                         betas=(self.cfg.beta1, self.cfg.beta2),
                                         eps=self.cfg.eps)
        elif self.cfg.optim == 'sgd':
            optimizer = torch.optim.SGD(param_groups, self.cfg.lr,
                                        momentum=self.cfg.momentum)
        else:
            raise NotImplementedError(self.cfg.optim)
        return optimizer

    def _create_lr_scheduler(self):
        return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.cfg.lr_decay_factor)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self._log.warning("Warning: There\'s no GPU available on this machine,"
                              "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self._log.warning(
                "Warning: The number of GPU\'s configured to use is {}, "
                "but only {} are available.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def save_model(self, error, name):
        is_best = error < self.best_error

        if is_best:
            self.best_error = error

        models = {'epoch': self.i_epoch,
                  'state_dict': self.model.module.state_dict()}

        save_checkpoint(self.save_root, models, name, is_best)
