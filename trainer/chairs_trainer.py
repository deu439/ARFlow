import time
import torch
from .base_trainer import BaseTrainer
from utils.flow_utils import evaluate_flow, torch_flow2rgb
from utils.misc_utils import AverageMeter
import matplotlib.pyplot as plt
import numpy as np


class TrainFramework(BaseTrainer):
    def __init__(self, train_loader, valid_loader, model, loss_func,
                 _log, save_root, config):
        super(TrainFramework, self).__init__(
            train_loader, valid_loader, model, loss_func, _log, save_root, config)

    def _run_one_epoch(self):
        am_batch_time = AverageMeter()
        am_data_time = AverageMeter()

        key_meter_names = ['Loss', 'l_ph', 'l_sm', 'flow_mean']
        key_meters = AverageMeter(i=len(key_meter_names), precision=4)

        self.model.train()
        end = time.time()

        if 'stage1' in self.cfg:
            if self.i_epoch == self.cfg.stage1.epoch:
                self.loss_func.cfg.update(self.cfg.stage1.loss)

        for i_step, data in enumerate(self.train_loader):
            if i_step > self.cfg.epoch_size:
                break

            # read data to device
            img1, img2 = data['img1'], data['img2']
            img_pair = torch.cat([img1, img2], 1).to(self.device)

            if 'img1_ph' in data and 'img2_ph' in data:
                img1_ph, img2_ph = data['img1_ph'], data['img2_ph']
                img_pair_ph = torch.cat([img1_ph, img2_ph], 1).to(self.device)
            else:
                img_pair_ph = img_pair

            # measure data loading time
            am_data_time.update(time.time() - end)

            # compute prediction, use photometric augmented images
            res_dict = self.model(img_pair_ph, with_bk=True)
            flows_12, flows_21 = res_dict['flows_fw'], res_dict['flows_bw']
            flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in
                     zip(flows_12, flows_21)]

            # Compute loss, use original images
            loss, l_ph, l_sm, flow_mean, mask = self.loss_func(flows, img_pair)

            # make sure loss does not contain NaNs
            assert (not np.isnan(loss.item())), "training loss is NaN"

            # update meters
            key_meters.update([loss.item(), l_ph.item(), l_sm.item(), flow_mean.item()],
                              img_pair.size(0))

            # compute gradient and do optimization step
            self.optimizer.zero_grad()
            loss.backward()

            #scaled_loss = 1024. * loss
            #scaled_loss.backward()

            #for param in [p for p in self.model.parameters() if p.requires_grad]:
            #    param.grad.data.mul_(1. / 1024)

            self.optimizer.step()

            # measure elapsed time
            am_batch_time.update(time.time() - end)
            end = time.time()

            if self.i_iter % self.cfg.record_freq == 0:
                for v, name in zip(key_meters.val, key_meter_names):
                    self.summary_writer.add_scalar('Train_' + name, v, self.i_iter)

            if self.i_iter % self.cfg.print_freq == 0:
                istr = '{}:{:04d}/{:04d}'.format(
                    self.i_epoch, i_step, self.cfg.epoch_size) + \
                       ' Time {} Data {}'.format(am_batch_time, am_data_time) + \
                       ' Info {}'.format(key_meters)
                self._log.info(istr)

            self.i_iter += 1
        self.i_epoch += 1

    @torch.no_grad()
    def _validate_with_gt(self):
        batch_time = AverageMeter()

        if type(self.valid_loader) is not list:
            self.valid_loader = [self.valid_loader]

        # only use the first GPU to run validation, multiple GPUs might raise error.
        # https://github.com/Eromera/erfnet_pytorch/issues/2#issuecomment-486142360
        self.model = self.model.module
        self.model.eval()

        end = time.time()

        all_error_names = []
        all_error_avgs = []

        n_step = 0
        for i_set, loader in enumerate(self.valid_loader):
            error_names = ['EPE']
            if hasattr(self.cfg, 'valid_masks') and self.cfg.valid_masks:
                error_names += ['E_noc', 'E_occ', 'F1_all']
            error_meters = AverageMeter(i=len(error_names))
            for i_step, data in enumerate(loader):
                img1, img2 = data['img1'], data['img2']
                img_pair = torch.cat([img1, img2], 1).to(self.device)
                gt_flows = data['target']['flow'].numpy().transpose([0, 2, 3, 1])

                # compute output
                res_dict = self.model(img_pair)

                # Evaluate loss
                flows_12, flows_21 = res_dict['flows_fw'], res_dict['flows_bw']
                flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in
                         zip(flows_12, flows_21)]
                loss, l_ph, l_sm, flow_mean, mask = self.loss_func(flows, img_pair)

                # Evaluate prediction
                pred_flows = flows_12[0].detach().cpu().numpy().transpose([0, 2, 3, 1])
                es = evaluate_flow(gt_flows, pred_flows)
                error_meters.update([l.item() for l in es], img_pair.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i_step % self.cfg.print_freq == 0 or i_step == len(loader) - 1:
                    self._log.info('Test: {0}[{1}/{2}]\t Time {3}\t '.format(
                        i_set, i_step, self.cfg.valid_size, batch_time) + ' '.join(
                        map('{:.2f}'.format, error_meters.avg)))

                if i_step > self.cfg.valid_size:
                    break
            n_step += len(loader)

            # write error to tf board.
            for value, name in zip(error_meters.avg, error_names):
                self.summary_writer.add_scalar(
                    'Valid_{}_{}'.format(name, i_set), value, self.i_epoch)

            # write predicted and true flow to tf board
            gt_flow = data['target']['flow']
            image = torch_flow2rgb(gt_flow[:, :2].cpu())
            self.summary_writer.add_images(f"Valid/gt", image, self.i_epoch)
            image = torch_flow2rgb(flows_12[0].cpu())
            self.summary_writer.add_images("Valid/pred_{}".format(i_set), image, self.i_epoch)
            self.summary_writer.add_image("Valid/mask_{}".format(i_set), mask.cpu(), self.i_epoch, dataformats='NCHW')

            all_error_avgs.extend(error_meters.avg)
            all_error_names.extend(['{}_{}'.format(name, i_set) for name in error_names])

        self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        # In order to reduce the space occupied during debugging,
        # only the model with more than cfg.save_iter iterations will be saved.
        if self.i_iter > self.cfg.save_iter:
            self.save_model(all_error_avgs[0], name='Chairs')

        return all_error_avgs, all_error_names
