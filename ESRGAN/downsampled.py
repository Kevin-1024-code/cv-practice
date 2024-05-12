import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import matplotlib.pyplot as plt

# 加载图像

image = Image.open('LR/woman_mini_d4_gaussian.png')


# 使用bicubic插值进行下采样
downsampled = image.resize((image.width // 4, image.height // 4), Image.BICUBIC)

# 使用bicubic插值进行上采样，作为超分辨率的一个简单示例
upsampled = downsampled.resize(image.size, Image.BICUBIC)

downsampled.save('LR/bicubic/woman_mini_d4_gaussian.png')

plt.imshow(np.array(upsampled))
plt.show()

print(np.array(image).shape)



# 计算PSNR和SSIM
psnr_value = psnr(np.array(image), np.array(upsampled))
ssim_value = ssim(np.array(image), np.array(upsampled),channel_axis=2)

print('PSNR: ', psnr_value)
print('SSIM: ', ssim_value)


