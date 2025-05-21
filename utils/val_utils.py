import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import torch.nn.functional as F

def compute_psnr_ssim(recoverd, clean):
    assert recoverd.shape == clean.shape
    recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1)

    recoverd = recoverd.transpose(0, 2, 3, 1)
    clean = clean.transpose(0, 2, 3, 1)
    psnr = 0
    ssim = 0

    for i in range(recoverd.shape[0]):
        # psnr_val += compare_psnr(clean[i], recoverd[i])
        # ssim += compare_ssim(clean[i], recoverd[i], multichannel=True)
        psnr += peak_signal_noise_ratio(clean[i], recoverd[i], data_range=1)
        ssim += structural_similarity(clean[i], recoverd[i], data_range=1, channel_axis=2)

    return psnr / recoverd.shape[0], ssim / recoverd.shape[0], recoverd.shape[0]

def pad_to_multiple_of_14(img: torch.Tensor) -> torch.Tensor:
    """
    Pads a tensor [B, C, H, W] so that H and W are multiples of 14.
    """
    B, C, H, W = img.shape
    pad_h = (14 - H % 14) % 14
    pad_w = (14 - W % 14) % 14

    # Pad at the bottom and right
    padded_img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')  # or 'constant'
    return padded_img