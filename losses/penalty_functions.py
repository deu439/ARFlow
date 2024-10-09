import torch

def abs_robust_loss(diff, eps=0.01, q=0.4):
    return torch.pow((torch.abs(diff) + eps), q)

def charbonnier(x_sq, eps=0.001):
    if type(x_sq) is list:
        x_sq = sum(x_sq)

    return torch.sqrt(x_sq + eps**2)

def charbonnier_prime(x_sq, eps=0.001):
    if type(x_sq) is list:
        x_sq = sum(x_sq)

    return 1 / (2 * torch.sqrt(x_sq + eps**2))

def identity(x):
    return x

def identity_prime(x):
    return torch.ones_like(x)

def get_penalty(name, derivative=False):
    if name == 'identity':
        return identity_prime if derivative else identity

    elif name == 'charbonnier':
        return charbonnier_prime if derivative else charbonnier

    elif name == 'abs_robust_loss':
        if derivative:
            raise NotImplementedError("derivative not implemented ofr abs_robust_loss penalty!")
        else:
            return abs_robust_loss