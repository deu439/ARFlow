import numbers
import random
import torch
import torch.nn.functional as F


def get_geometric_transforms(cfg):
    transforms = []
    if hasattr(cfg, 'crop') and cfg.crop:
        transforms.append(RandomCrop(cfg.crop_size))
    if hasattr(cfg, 'hflip') and cfg.hflip:
        transforms.append(RandomHorizontalFlip())
    if hasattr(cfg, 'scale') and cfg.scale:
        transforms.append(Scale(cfg.scale_size))
    return Compose(transforms)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input):
        for t in self.transforms:
            input = t(input)
        return input


class RandomCrop(object):
    """
    Crops the given (C x H x W) tensor at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs):
        h, w = inputs[0].shape[-2:]
        th, tw = self.size
        if w == tw and h == th:
            return inputs

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        inputs = inputs[..., y1 : y1 + th, x1 : x1 + tw]
        return inputs


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, inputs):
        if random.random() < 0.5:
            inputs = torch.flip(inputs, dims=[-1])
        return inputs

class Scale(object):
    """
    Deterministic scaling
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, inputs):
        return F.interpolate(inputs, size=self.size, mode='bilinear', align_corners=False)
