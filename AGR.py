import torch


def AGR(x):
    grad = x.grad
    grad_sum = torch.abs(grad).sum(dim=tuple(range(0,len(grad.size()))),keepdim=True)
    x.grad = grad - torch.mul(grad,torch.abs(grad)/grad_sum)
    return x   