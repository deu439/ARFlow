import time
import torch
from .base_trainer import BaseTrainer
from utils.flow_utils import evaluate_flow, torch_flow2rgb, evaluate_uncertainty, CalibrationCurve
from utils.misc_utils import AverageMeter, matplot_fig_to_numpy, mixture_entropy
import utils.uflow_utils as uflow_utils
#from triag_solve_cuda import inverse_diagonal

import matplotlib.pyplot as plt
import numpy as np
import PIL
import math


class TrainFramework(BaseTrainer):
    def __init__(self, train_loader, valid_loader, model, loss_func,
                 _log, save_root, config):
        super(TrainFramework, self).__init__(
            train_loader, valid_loader, model, loss_func, _log, save_root, config)

    def _run_one_epoch(self):
        am_batch_time = AverageMeter()
        am_data_time = AverageMeter()

        key_meter_names = ['Loss', 'l_ph', 'l_sm', 'entropy', 'l_oof']
        key_meters = AverageMeter(i=len(key_meter_names), precision=4)

        self.model.train()
        end = time.time()

        # Initialize model and optimizer state
        #model_state = copy.deepcopy(self.model.state_dict())
        #optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        low_lr = 0

        if 'stage1' in self.cfg:
            if self.i_epoch == self.cfg.stage1.epoch:
                self.loss_func.cfg.update(self.cfg.stage1.loss)

        for i_step, data in enumerate(self.train_loader):
            if i_step > self.cfg.epoch_size:
                break
            # read data to device
            img1, img2 = data['img1'].to(self.device), data['img2'].to(self.device)
            #img_pair = torch.cat([img1, img2], 1).to(self.device)

            # measure data loading time
            am_data_time.update(time.time() - end)

            # compute output
            res_dict = self.model(img1, img2, with_bk=True)
            #flows_12, flows_21 = res_dict['flows_fw'], res_dict['flows_bw']
            #flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in
            #         zip(flows_12, flows_21)]
            loss, l_ph, l_sm, entropy, l_oof, _, _, _ = self.loss_func(res_dict, img1, img2)

            # update meters
            key_meters.update([loss.item(), l_ph.item(), l_sm.item(), entropy.item(), l_oof], img1.size(0))

            # If large l1norm is detected, use lower learning rate
            # if inv_l1norm > 100.0:
            #     if low_lr == 0:
            #         self._log.info("inv_l1norm > 100.0, setting learning rate to 1e-6.")
            #         for g in self.optimizer.param_groups:
            #             g['lr'] = 1e-6
            #
            #     low_lr = 100
            #
            # if low_lr > 1:
            #     low_lr = low_lr - 1
            # elif low_lr == 1:
            #     self._log.info("inv_l1norm < 10.0, setting learning rate back to {}".format(self.cfg.lr))
            #    for g in self.optimizer.param_groups:
            #        g['lr'] = self.cfg.lr
            #    low_lr = 0

            # if nan is encountered, revert the last step and continue training
            #if np.isnan(loss.item()):
            #    self.model.load_state_dict(model_state)
            #    self.optimizer.__setstate__({'state': defaultdict(dict)})
            #    #self.optimizer.load_state_dict(optimizer_state)
            #    continue

            # store model and optimizer state before making the next step
            #model_state = copy.deepcopy(self.model.state_dict())
            #optimizer_state = copy.deepcopy(self.optimizer.state_dict())

            # compute gradient and do optimization step
            self.optimizer.zero_grad()

            # backpropagation
            loss.backward()

            # Gradient clipping
            if self.cfg.clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip)

            # update parameters
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
        if self.cfg.track_auc:
            cc = CalibrationCurve()
        for i_set, loader in enumerate(self.valid_loader):
            error_names = ['Loss', 'l_ph', 'l_sm', 'entropy', 'l_oof', 'EPE']
            if hasattr(self.cfg, 'valid_masks') and self.cfg.valid_masks:
                error_names += ['E_noc', 'E_occ', 'F1_all']
            if hasattr(self.cfg, 'track_auc') and self.cfg.track_auc:
                error_names += ['AUC', 'AUC_diff']

            error_meters = AverageMeter(i=len(error_names))
            splots = []
            oplots = []

            for i_step, data in enumerate(loader):
                img1, img2 = data['img1'].to(self.device), data['img2'].to(self.device)
                #img_pair = torch.cat([img1, img2], 1).to(self.device)
                gt_flows = data['target']['flow'].numpy().transpose([0, 2, 3, 1])

                # Compute output
                res_dict = self.model(img1, img2)

                # Evaluate loss
                #flows_12, flows_21 = res_dict['flows_fw'], res_dict['flows_bw']
                #flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in
                #         zip(flows_12, flows_21)]
                loss, l_ph, l_sm, entropy, l_oof, sample_flows, occu_mask, valid_mask  = self.loss_func(res_dict, img1, img2)
                error_values = [loss, l_ph, l_sm, entropy, l_oof]

                # Evaluate endpoint error
                flows = res_dict['flows_fw']
                pred_flows = flows[0][:, 0:2].detach().cpu().numpy().transpose([0, 2, 3, 1])
                es = evaluate_flow(gt_flows, pred_flows)
                error_values += es

                # Compute entropy of the horizontal and vertical component. If we track AUC this needs to be computed
                # for each batch, otherwise we compute it only for the last batch.
                if self.cfg.track_auc or i_step == len(loader) - 1:
                    if self.loss_func.cfg.approx == 'diag':
                        uv_entropy = flows[0][:, 2:4]

                    elif self.loss_func.cfg.approx == 'mixture':
                        K = self.loss_func.cfg.n_components
                        mean = flows[0][:, 0:K*2]
                        logstd = flows[0][:, K*2:K*2+2]
                        uv_entropy = mixture_entropy(mean, logstd, n_samples=100)

                    elif self.loss_func.cfg.approx == 'sparse':
                        if self.loss_func.cfg.inv_cov:
                            log_diag = flows[2][:, 2:4]
                            left = flows[2][:, 4:6, :, :-1]
                            over = flows[2][:, 6:8, :-1, :]
                            uv_entropy = inverse_diagonal(torch.exp(log_diag).contiguous(), left.contiguous(), over.contiguous())
                            uv_entropy = uflow_utils.upsample(uv_entropy + 2*math.log(4), scale_factor=4, is_flow=False, align_corners=False)
                        else:
                            uv_entropy = flows[0][:, 2:4]

                    elif self.loss_func.cfg.approx == 'lowrank':
                        std = flows[2][:, 2:2 + 2 * self.loss_func.cfg.columns]
                        std_u = std[:, 0::2]
                        std_v = std[:, 1::2]
                        u_entropy = torch.log(torch.sum(std_u * std_u, dim=1, keepdim=True)) / 2
                        v_entropy = torch.log(torch.sum(std_v * std_v, dim=1, keepdim=True)) / 2
                        uv_entropy = torch.concat((u_entropy, v_entropy), dim=1)
                        uv_entropy = uflow_utils.upsample(uv_entropy + 2 * math.log(4), scale_factor=4, is_flow=False, align_corners=False)

                # Evaluate AUC if needed
                if self.cfg.track_auc:
                    uv_entropy_np = uv_entropy.detach().cpu().numpy().transpose([0, 2, 3, 1])
                    auc, splot, oplot = evaluate_uncertainty(gt_flows, pred_flows, uv_entropy_np, sp_samples=self.cfg.sp_samples)
                    splots += splot
                    oplots += oplot
                    error_values += auc
                    
                    cc(gt_flows=gt_flows, pred_flows=pred_flows, pred_entropies=uv_entropy_np)

                # Update error meters
                error_meters.update(error_values, img1.size(0))

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

            # Record output tensor
            torch.save(flows[2], self.save_root / 'flow_fw_l2_{}.pt'.format(self.i_epoch))

            # Write predicted and true flow to tboard
            gt_flow = data['target']['flow']
            image = torch_flow2rgb(gt_flow[:, :2].cpu())
            self.summary_writer.add_images("Valid/gt_{}".format(i_set), image, self.i_epoch)

            n_components = self.loss_func.cfg.n_components
            for k in range(n_components):
                image = torch_flow2rgb(flows[0][:, 2*k:2*(k+1)].detach().cpu())
                image_np = image.numpy().transpose(0, 2, 3, 1)
                image_np = (image_np * 255.0).astype(np.uint8)

                # Print weight into the prediction
                if 'weights_fw' in res_dict:
                    for l in range(image.size(0)):
                        weight = res_dict['weights_fw'][l, k].item()
                        pimg = PIL.Image.fromarray(image_np[l])
                        font = PIL.ImageFont.truetype('utils/DejaVuSansMono.ttf', 16)
                        PIL.ImageDraw.Draw(pimg).text((4, 4), "{:.2f}".format(weight), (0, 0, 0), font)
                        image_np[l] = np.array(pimg)

                self.summary_writer.add_images("Valid/pred_{}_{}".format(i_set, k), image_np, self.i_epoch, dataformats='NHWC')

            # Write entropy to tboard
            entropy = torch.sum(uv_entropy, dim=1, keepdim=True)
            entropy -= torch.min(entropy)
            entropy /= torch.max(entropy)
            self.summary_writer.add_images("Valid/entropy_{}_{}".format(i_set, k), entropy.cpu(), self.i_epoch, dataformats='NCHW')

            # Write sparsification plots to tboard
            if len(splots) > 0 and len(oplots) > 0:
                x_axis = np.linspace(0, 1, self.cfg.sp_samples)
                y_axis1 = np.mean(splots, axis=0)
                y_axis2 = np.mean(oplots, axis=0)
                fig, ax = plt.subplots()
                ax.plot(x_axis, y_axis1)
                ax.plot(x_axis, y_axis2)
                ax.legend(['splot', 'oracle'])
                np_fig = matplot_fig_to_numpy(fig)
                self.summary_writer.add_image("Valid/splot_{}".format(i_set), np_fig, self.i_epoch, dataformats="HWC")

            # Write sample images and the corresponding pixel weights
            sample_flows_image = torch_flow2rgb(sample_flows.detach().cpu())
            self.summary_writer.add_image("Valid/sample_flows_{}".format(i_set), sample_flows_image.cpu(), self.i_epoch, dataformats='NCHW')
            self.summary_writer.add_image("Valid/occu_masks_{}".format(i_set), occu_mask.cpu(), self.i_epoch, dataformats='NCHW')
            self.summary_writer.add_image("Valid/valid_masks_{}".format(i_set), valid_mask.cpu(), self.i_epoch, dataformats='NCHW')

            all_error_avgs.extend(error_meters.avg)
            all_error_names.extend(['{}_{}'.format(name, i_set) for name in error_names])

        
        vals, means, sigmas = cc.calibration_curve()
        plt.figure()
        plt.errorbar(vals, means, sigmas, fmt='o', linewidth=2, capsize=6)
        plt.xlabel('sigma')
        plt.ylabel('epe')
        plt.grid()
        plt.savefig('foo.png')
        np.save("foo.npy", (vals, means, sigmas))
        
        self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        # In order to reduce the space occupied during debugging,
        # only the model with more than cfg.save_iter iterations will be saved.
        if self.i_iter > self.cfg.save_iter:
            self.save_model(all_error_avgs[0], name='Chairs')

        return all_error_avgs, all_error_names
