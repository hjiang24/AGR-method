import torch

weights = torch.load('output/TinyViT-5M-1k/default/ckpt_epoch_299.pth')
print(weights)