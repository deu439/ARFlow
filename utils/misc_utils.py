import collections
import io
import numpy as np
import torch


def update_dict(orig_dict, new_dict):
    for key, val in new_dict.items():
        if isinstance(val, collections.Mapping):
            tmp = update_dict(orig_dict.get(key, {}), val)
            orig_dict[key] = tmp
        else:
            orig_dict[key] = val
    return orig_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, i=1, precision=3, names=None):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)
        self.names = names
        if names is not None:
            assert self.meters == len(self.names)
        else:
            self.names = [''] * self.meters

    def reset(self, i):
        self.val = [0] * i
        self.avg = [0] * i
        self.sum = [0] * i
        self.count = [0] * i

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        if not isinstance(n, list):
            n = [n] * self.meters
        assert (len(val) == self.meters and len(n) == self.meters)
        for i in range(self.meters):
            self.count[i] += n[i]
        for i, v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n[i]
            self.avg[i] = self.sum[i] / self.count[i]

    def __repr__(self):
        val = ' '.join(['{} {:.{}f}'.format(n, v, self.precision) for n, v in
                        zip(self.names, self.val)])
        avg = ' '.join(['{} {:.{}f}'.format(n, a, self.precision) for n, a in
                        zip(self.names, self.avg)])
        return '{} ({})'.format(val, avg)


def matplot_fig_to_numpy(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im


def log_sum_exp(x, w=1, dim=0):
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    return x_max + torch.log(torch.sum(w * torch.exp(x - x_max), dim=dim, keepdim=True))


def gaussian_mixture_log_pdf(flow, mean, log_std, weights, per_pixel=False):
    nsamples = flow.size(0) // mean.size(0)
    mean = mean.repeat(nsamples, 1, 1, 1)
    log_std = log_std.repeat(nsamples, 1, 1, 1)
    weights = weights.repeat(nsamples, 1)
    std = torch.exp(log_std)

    # vertical component
    u_err = (flow[:, [0]] - mean[:, 0::2]) / std[:, 0::2]    # (nsamples * batch_size, K, rows, cols)
    u_err_sq = u_err*u_err
    log_det_u = log_std[:, 0::2]         # (nsamples * batch_size, K, rows, cols)

    # horizontal component
    v_err = (flow[:, [1]] - mean[:, 1::2]) / std[:, 1::2]  # (nsamples * batch_size, K, rows, cols)
    v_err_sq = v_err*v_err
    log_det_v = log_std[:, 1::2]         # (nsamples * batch_size, K, rows, cols)

    err_sq = u_err_sq + v_err_sq
    log_det = log_det_u + log_det_v

    if per_pixel:
        log_pdf = log_sum_exp(-log_det - err_sq / 2, weights[:, :, None, None], dim=1)

    else:
        err_sq = torch.sum(err_sq, dim=(2, 3))              # (nsamples * batch_size, K)
        log_det = torch.sum(log_det, dim=(2, 3))            # (nsamples * batch_size, K)
        rows, cols = flow.shape[2:]
        log_pdf = log_sum_exp(-log_det - err_sq / 2, weights, dim=1) / (rows * cols)

    return log_pdf  # (nsamples * batch_size, 1) or (nsamples*batch_size, 1, rows, cols)


def mixture_entropy(mean, log_std, weights, n_samples=100):
    Normal = torch.distributions.Normal(0, 1)
    #if torch.cuda.is_available():
    #    Normal.loc = Normal.loc.cuda()  # hack to get sampling on the GPU
    #    Normal.scale = Normal.scale.cuda()

    def sample(mean, std, weights):
        nsamples = 1
        rows, cols = mean.shape[2:]
        z = torch.multinomial(weights, num_samples=nsamples, replacement=True)      # (batch, nsamples)
        z = z[:, :, None, None].repeat(1, 1, rows, cols)                            # (batch, nsamples, rows, cols)
        # Batches change fast, samples slow (to be consistent with the diag array)
        mean_u = torch.gather(mean, 1, 2*z).transpose(1,0).reshape(-1, 1, rows, cols)    # (batch*nsamples, 1, rows, cols)
        std_u = torch.gather(std, 1, 2*z).transpose(1,0).reshape(-1, 1, rows, cols)      # (batch*nsamples, 1, rows, cols)
        mean_v = torch.gather(mean, 1, 2*z+1).transpose(1,0).reshape(-1, 1, rows, cols)  # (batch*nsamples, 1, rows, cols)
        std_v = torch.gather(std, 1, 2*z+1).transpose(1,0).reshape(-1, 1, rows, cols)      # (batch*nsamples, 1, rows, cols)
        mean = torch.cat((mean_u, mean_v), dim=1)
        std = torch.cat((std_u, std_v), dim=1)
        z = mean + std * Normal.sample(std.size())
        return z

    std = torch.exp(log_std)
    batch, _, row, col = mean.shape
    entropy = torch.zeros((batch, 1, row, col))
    for i in range(n_samples):
        flow = sample(mean, std, weights)
        entropy -= gaussian_mixture_log_pdf(flow, mean, log_std, weights, per_pixel=True)

    return entropy / n_samples

