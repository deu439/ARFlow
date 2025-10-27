import torch
import math

from datasets.get_dataset import get_dataset
from losses.get_loss import get_loss
from utils.uflow_utils import downsample

from easydict import EasyDict
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar
from losses.uflow_elbo_loss import data_loss_no_penalty, smooth_loss_no_penalty

cfg = {
    "data": [{
        "root_chairs": "/home/deu/Datasets/FlyingChairs2/",
        "run_at": False,
        "test_shape": [384, 512],
        "train_n_frames": 2,
        "type": "Chairs",
    }],
    "data_aug": {
        "crop": False,
        "hflip": False,
        "swap": False
    },

    "loss": {
        "edge_constant": 150,
        "edge_asymp": 0.01,
        "type": "uflow_elbo",
        "w_smooth": 4.0,
        "data_loss": ["ssim"],
        "data_weight": [1.0],
        "data_penalty": ["identity"],
        "w_entropy": 0.1,
        "with_bk": True,
        "align_corners": False,
        "diag": False,
        "diag_dominant": False,
        "inv_cov": True,
        "approx_entropy": False,
        "occu_mean": False,
        "n_samples": 1,
        "offdiag_reg": 0.0,
        "align_corners": False,
    },

    "train": {
        #"penalty": "data",
        "penalty": "smooth",
        "init_vars": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50],
        #"init_vars": [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
        "subsample": 0.95,
        "n_samples": 3e6,
        "batch_size": 4,
        "workers": 4,
        "n_iter": 30
    },
}


def gaussian_mixture(x, pi, mu, beta):
    arg = -beta[None, :] * (x[:, None] - mu[None, :]) ** 2
    w = pi * np.sqrt(beta) / np.sqrt(2 * np.pi)
    return np.sum(w[None, :] * np.exp(arg/2), axis=1)


def robust_l1(x, eps=0.001):

    """Robust L1 metric."""
    return np.exp(-(x ** 2 + eps ** 2) ** 0.5) / 2


def robust_l1_fwhm(eps=0.001):
    return 2*np.sqrt((eps + np.log(2))**2 - eps**2)


def abs_robust_loss(diff, eps=0.01, q=0.4):
    return np.exp(-np.power((np.abs(diff) + eps), q)) / 6.6288


def abs_robust_loss_fwhm(eps=0.01, q=0.4):
    return 2*(np.power(eps**q + np.log(2), 1/q) - eps)

class EM:
    def __init__(self, k=10, init_vars=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 5, 10, 100, 1000]):
        # Prior parameters
        self.k = k
        self.alpha = torch.ones(k)
        self.mu_0 = 0
        self.beta_0 = 1e-3
        self.a = 1
        self.b = 1

        # Variational parameters
        self.pi = torch.ones(k) / k
        self.mu = torch.zeros(k)
        self.beta = 1 / torch.tensor(init_vars)
        self.alpha_bar = self.alpha.clone()
        self.xi = None

    def update_xi(self, x):
        """
        Update variational parameters xi
        :param x: list [x0, x1]; x0 is m vector of samples, x1 is m vector of weights
        :param mu: k vector of component means
        :param beta: k vector of component inverse variances
        :param alpha_bar: k vector of parameters of the Dirichlet posterior of pi
        :return: xi: m x k array of variational parameters
        """
        x0 = x[0]
        log_pi = torch.special.digamma(self.alpha_bar) - torch.special.digamma(torch.sum(self.alpha_bar, dim=0, keepdim=True))
        arg = -self.beta[None, :] * ((x0[:, None] - self.mu[None, :])**2) / 2 + log_pi[None, :]
        w = torch.sqrt(self.beta)
        num = w[None, :] * torch.exp(arg - torch.max(arg, dim=1, keepdim=True).values)
        den = torch.sum(num, dim=1, keepdim=True)

        self.xi = num / den

    def update_pi(self, x):
        """
        Update mixture component weights pi
        :param x: list [x0, x1]; x0 is m vector of samples, x1 is m vector of weights
        :param xi: m x k array of variational parameters
        :param alpha: k vector of Dirichlet parameters
        :return: pi: k vector of mixture component weights
        :return: alpha_bar: k vector of posterior Dirichlet parameters of p(pi)
        """

        x1 = x[1]
        xi_sum = torch.sum(x1[:, None] * self.xi, dim=0)
        alpha_bar = self.alpha + xi_sum
        den = torch.sum(alpha_bar, dim=0, keepdim=True)

        self.pi = alpha_bar / den
        self.alpha_bar = alpha_bar

    def update_mu_ml(self, x):
        """
        Update component means mu
        :param x: list [x0, x1]; x0 is m vector of samples, x1 is m vector of weights
        :param xi: m x k array of variational parameters
        :return: mu: k vector of mixture component means
        """
        x0 = x[0]
        x1 = x[1]
        num = torch.sum(self.xi * x1[:, None] * x0[:, None], dim=0)
        den = torch.sum(self.xi * x1[:, None], dim=0)

        self.mu = num / den

    def update_mu_map(self, x):
        """
        Update component means mu
        :param x: list [x0, x1]; x0 is m vector of samples, x1 is m vector of weights
        :param xi: m x k array of variational parameters
        :param mu_0: prior mean
        :param beta_0: prior precision
        :return: mu: k vector of mixture component means
        """
        x0 = x[0]
        x1 = x[1]
        num = self.beta_0*self.mu_0 + torch.sum(self.xi * x1[:, None] * x0[:, None], dim=0)
        den = self.beta_0 + torch.sum(self.xi * x1[:, None], dim=0)

        self.mu = num / den

    def update_beta_map(self, x):
        """
        Update component means mu
        :param x: list [x0, x1]; x0 is m vector of samples, x1 is m vector of weights
        :param xi: m x k array of variational parameters
        :param: mu: k vector of mixture component means
        :param mu_0: prior mean
        :param beta_0: prior precision
        :return: beta: k vector of mixture component inverse variances
        """
        x0 = x[0]
        x1 = x[1]
        num = 2*self.a - 1 + torch.sum(self.xi * x1[:, None], dim=0)
        den = 2*self.b + self.beta_0*(self.mu - self.mu_0)**2 + torch.sum(self.xi * x1[:, None] * (x0[:, None] - self.mu[None, :])**2, dim=0)
        self.beta = num / den

    def objective(self, x):
        """
        :param x: list [x0, x1]; x0 is m vector of samples, x1 is m vector of weights
        :param xi: m x k array of variational parameters
        :param: mu: k vector of mixture component means
        :param beta: k vector of component inverse variances
        :param alpha_bar: k vector of parameters of the Dirichlet posterior of pi
        """
        x0 = x[0]
        x1 = x[1]
        # Summation over data - i
        sum_i = torch.sum(self.xi * x1[:, None] * (torch.log(self.beta)[None, :] - math.log(2*torch.pi)
                          - self.beta[None, :] * (x0[:, None] - self.mu[None, :])**2)/2
                          - x1[:, None] * torch.xlogy(self.xi, self.xi), dim=0)

        # Summation over components - j
        sum_j = torch.sum((self.a - 1/2)*torch.log(self.beta) - (self.beta_0*self.beta*(self.mu-self.mu_0)**2)/2
                       - self.b*self.beta + sum_i)

        # Log of the integral
        log_integral = torch.sum(torch.lgamma(self.alpha_bar)) - torch.lgamma(torch.sum(self.alpha_bar))

        return sum_j + log_integral

    def update(self, x):
        """
        :param x: list [x0, x1]; x0 is m vector of samples, x1 is m vector of weights
        :return:
        """

        # Run EM updates
        self.update_xi(x)
        self.update_pi(x)
        self.update_beta_map(x)

        return self.objective(x)


if __name__ == "__main__":
    cfg = EasyDict(cfg)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Get dataset and loader
    train_set, _ = get_dataset(cfg)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.train.batch_size, num_workers=cfg.train.workers, pin_memory=True, shuffle=True)

    # Get loss
    loss_funct = get_loss(cfg.loss)

    # Collect samples
    data_list = []
    for i_step, data in enumerate(train_loader):
        num_samples = sum([x.size(-1) for x in data_list])
        print("Batch: {:d}, Samples:{:d}k".format(i_step, num_samples//1000))
        if num_samples > cfg.train.n_samples:
            break

        # Get data
        im1_0, im2_0 = data['img1'], data['img2']
        flows12_0 = data['target']['flow']
        flows21_0 = data['target']['flow_bw']

        # Subsample loss (inference is done at 1/4 resolution)
        flows12_2 = downsample(flows12_0, scale_factor=4, align_corners=cfg.loss.align_corners, is_flow=True)
        flows21_2 = downsample(flows21_0, scale_factor=4, align_corners=cfg.loss.align_corners, is_flow=True)

        # Calculate data loss
        if cfg.train.penalty == 'data':
            data_pixel_loss12, data_pixel_weight12 = data_loss_no_penalty(im1_0, im2_0, flows12_2, flows21_2)
            loss_list = [data_pixel_loss12[0]]
            weight_list = [data_pixel_weight12[0]]
            if cfg.loss.with_bk:
                # Arguments are passed in reverse order!
                data_pixel_loss21, data_pixel_weight21 = data_loss_no_penalty(im2_0, im1_0, flows21_2, flows12_2)
                loss_list += [data_pixel_loss21[0], ]
                weight_list += [data_pixel_weight21[0], ]

        else:
            # Calculate smoothness loss
            smooth_loss12_x, smooth_weight12_x, smooth_loss12_y, smooth_weight12_y = smooth_loss_no_penalty(
                im1_0, flows12_2, cfg.loss.align_corners, cfg.loss.edge_constant, cfg.loss.edge_asymp
            )
            loss_list = [smooth_loss12_x[:, :, :, 0:-1], smooth_loss12_y[:, :, 0:-1, :]]
            weight_list = [smooth_weight12_x[:, :, :, 0:-1].repeat([1, 2, 1, 1]),
                           smooth_weight12_y[:, :, 0:-1, :].repeat([1, 2, 1, 1])]
            if cfg.loss.with_bk:
                # Arguments are passed in reverse order!
                smooth_loss21_x, smooth_weight21_x, smooth_loss21_y, smooth_weight21_y = smooth_loss_no_penalty(
                    im2_0, flows21_2, cfg.loss.align_corners, cfg.loss.edge_constant, cfg.loss.edge_asymp
                )
                loss_list += [torch.mean(smooth_loss21_x, dim=-1), torch.mean(smooth_loss21_y, dim=-1)]
                weight_list += [smooth_weight21_x[:, :, :, 0:-1], smooth_weight21_y[:, :, 0:-1, :]]

        # Subsample
        for loss, weight in zip(loss_list, weight_list):
            weight = weight / torch.max(weight)  # Normalize the weights
            binary_mask = weight > 1e-6          # Drop samples with too small weight
            binary_mask = binary_mask * (torch.rand_like(weight) > cfg.train.subsample)
            #binary_mask = torch.rand_like(weight) > cfg.train.subsample
            x0 = torch.masked_select(loss, binary_mask)
            #x1 = torch.masked_select(weight, binary_mask)
            x1 = torch.ones_like(x0)
            x = torch.stack([x0, x1])
            data_list.append(x)

    # Train GMM
    obj = []
    em = EM(k=10, init_vars=cfg.train.init_vars)
    x = torch.cat(data_list, dim=-1)

    for j in range(cfg.train.n_iter):
        print("update: ", j)
        obj.append(em.update(x))

    # Collect and print parameters
    pi = em.pi.numpy()
    mu = em.mu.numpy()
    beta = em.beta.numpy()
    print("Pi: ", list(pi))
    print("Beta: ", list(beta))

    # Plot the results
    if cfg.train.penalty == 'data':
        reference_penalty = abs_robust_loss
        reference_fwhm = abs_robust_loss_fwhm()
        x = np.linspace(-30, 30, 8000)
        x_label = "Generalized Hamming distance"
    else:
        reference_penalty = robust_l1
        reference_fwhm = robust_l1_fwhm()
        x = np.linspace(-20, 20, 8000)
        x_label = "Differences"

    # Find FWHM of the mixture
    func = lambda a: gaussian_mixture(np.array([reference_fwhm/2]), pi, mu, a*beta) - gaussian_mixture(np.array([0]), pi, mu, a*beta)/2
    sol = root_scalar(func, method="bisect", bracket=[1e-6, 100])

    print("Scaling factor: ", sol.root)
    print("Beta scaled: ", list(beta * sol.root))
    y = gaussian_mixture(x, pi, mu, beta*sol.root)
    #y = gaussian_mixture(x, pi, mu, beta)
    yp = reference_penalty(x)
    fig, ax = plt.subplots(1,2)
    # pdf
    ax[0].plot(np.sqrt(sol.root)*x, y)
    ax[0].plot(np.sqrt(sol.root)*x, yp)
    ax[0].set_xlabel(x_label)
    ax[0].set_ylabel('Probability density')
    ax[0].legend(['Gaussian mixture', 'Reference'])
    # penalty
    ax[1].plot(np.sqrt(sol.root)*x, -np.log(y))
    ax[1].plot(np.sqrt(sol.root)*x, -np.log(yp))
    ax[1].set_xlabel(x_label)
    ax[1].set_ylabel('Penalty')
    ax[1].legend(['Gaussian mixture', 'Reference'])
    fig.suptitle("Result")

    fig, ax = plt.subplots()
    ax.plot(np.arange(0, cfg.train.n_iter), obj)
    fig.suptitle("Objective")
    plt.show()