import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import matplotlib.pyplot as plt

# 加载图像

image1 = Image.open('LR/baby_mini_d4_gaussian.png')
image2 = Image.open('results/baby_mini_d4_gaussian_rlt.png')

# 计算PSNR和SSIM
psnr_value = psnr(np.array(image1), np.array(image2))
ssim_value = ssim(np.array(image1), np.array(image2),channel_axis=2)

print('PSNR: ', psnr_value)
print('SSIM: ', ssim_value)


#channel_axis=2