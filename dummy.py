import torch
from model import DenoiseNet

model = DenoiseNet()

x_dummy = torch.randn(32, 1, 128)
print(x_dummy.shape)

x_denoised = model(x_dummy)
print(x_denoised.shape)