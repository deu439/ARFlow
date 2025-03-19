import torch
import matplotlib.pyplot as plt
from utils.triag_solve import matrix_vector_product, backward_substitution
from utils.flow_utils import torch_flow2rgb


def reparam_triag(mean, diag, left, over, leftover, nsamples=1):
    mean = mean.repeat(nsamples, 1, 1, 1)
    diag = diag.repeat(nsamples, 1, 1, 1)
    left = left.repeat(nsamples, 1, 1, 1)
    over = over.repeat(nsamples, 1, 1, 1)
    leftover = leftover.repeat(nsamples, 1, 1, 1)
    Normal = torch.distributions.Normal(0, 1)
    eps = Normal.sample(mean.size())
    z = mean + matrix_vector_product(diag, left, over, leftover, eps)
    return z

def reparam_triag_inv(mean, diag, left, over, leftover, nsamples=1):
    mean = mean.repeat(nsamples, 1, 1, 1)
    diag = diag.repeat(nsamples, 1, 1, 1)
    left = left.repeat(nsamples, 1, 1, 1)
    over = over.repeat(nsamples, 1, 1, 1)
    leftover = leftover.repeat(nsamples, 1, 1, 1)
    Normal = torch.distributions.Normal(0, 1)
    eps = Normal.sample(mean.size())
    z = mean + backward_substitution(diag, left, over, leftover, eps)
    return z


mean = torch.zeros((1, 2, 100, 100))
diag = torch.ones((1, 2, 100, 100))
left = -0.9*torch.ones((1, 2, 100, 100-1))
over = -0.9*torch.ones((1, 2, 100-1, 100))
leftover = 0.8*torch.ones((1, 2, 100-1, 100-1))

flow = reparam_triag_inv(mean, diag, left, over, leftover)
flow_img = torch_flow2rgb(flow.cpu())

# To numpy
flow_img = flow_img.cpu().numpy().transpose(0, 2, 3, 1)

fig, ax = plt.subplots()
ax.imshow(flow_img[0])
plt.show()
