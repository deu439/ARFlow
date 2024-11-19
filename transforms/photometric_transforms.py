import numpy as np
import torch
import torchvision
import random


def get_photometric_transforms(cfg):
    transforms = []
    brightness = cfg.brightness if hasattr(cfg, 'brightness') else 0
    contrast = cfg.contrast if hasattr(cfg, 'contrast') else 0
    saturation = cfg.saturation if hasattr(cfg, 'saturation') else 0
    hue = cfg.hue if hasattr(cfg, 'hue') else 0

    if any([brightness, contrast, saturation, hue]) > 0:
        transforms.append(
            torchvision.transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        )

    if hasattr(cfg, 'gamma') and cfg.gamma > 0:
        transforms.append(RandomGamma(min_gamma=0.7, max_gamma=1.5, clip_image=True))

    if hasattr(cfg, 'swap_channels') and cfg.swap_channels:
        transforms.append(RandomSwapChannels())

    return torchvision.transforms.Compose(transforms)


class RandomGamma():
    def __init__(self, min_gamma=0.7, max_gamma=1.5, clip_image=False):
        self._min_gamma = min_gamma
        self._max_gamma = max_gamma
        self._clip_image = clip_image

    @staticmethod
    def get_params(min_gamma, max_gamma):
        return np.random.uniform(min_gamma, max_gamma)

    @staticmethod
    def adjust_gamma(image, gamma, clip_image):
        adjusted = torch.pow(image, gamma)
        if clip_image:
            adjusted.clamp_(0.0, 1.0)
        return adjusted

    def __call__(self, image):
        gamma = self.get_params(self._min_gamma, self._max_gamma)
        return self.adjust_gamma(image, gamma, self._clip_image)


class RandomSwapChannels():
    def __call__(selfs, image):
        ind = torch.randperm(image.shape[-3])
        return image[..., ind, :, :]


