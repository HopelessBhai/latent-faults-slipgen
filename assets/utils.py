# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
import json
from skimage.metrics import structural_similarity as sk_ssim      # SSIM
# import piq                                                        # VIF-P
from scipy.stats import wasserstein_distance, entropy                   # 1-D EMD


def plot_losses(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def l1_regularization(model, lambda_l1):
    """
    Computes L1 regularization loss for the model parameters.
    """
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return lambda_l1 * l1_norm


def gaussian(window_size, sigma):
    gauss = torch.tensor([np.exp(-(x - window_size//2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    # Create a 1D Gaussian window and then compute the outer product to get a 2D window.
    _1D_window = gaussian(window_size, sigma=1.5).unsqueeze(1)  # shape: (window_size, 1)
    _2D_window = _1D_window.mm(_1D_window.t()).float()  # shape: (window_size, window_size)
    window = _2D_window.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, window_size, window_size)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    # Ensure that images have the same number of channels
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)
    
    # Compute local means via convolution
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        # Mean SSIM over the channels and spatial dimensions for each image in the batch.
        return ssim_map.mean(1).mean(1).mean(1)

def ssim_loss(img1, img2, window_size=11, size_average=True):
    # Loss is defined as 1 minus the SSIM index.
    return 1 - ssim(img1, img2, window_size, size_average)


def clip_contrastive_loss(text_embeds, image_embeds, temperature=0.07):
    # Normalize
    text_embeds = nn.functional.normalize(text_embeds, dim=1)
    image_embeds = nn.functional.normalize(image_embeds, dim=1)
    
    # Cosine similarity
    logits_per_text = text_embeds @ image_embeds.T
    logits_per_image = image_embeds @ text_embeds.T
    # logits_per_image = logits_per_text.T
    
    # Scale by temperature
    logits_per_text /= temperature
    logits_per_image /= temperature

    # Labels (i-th pair is the correct one)
    batch_size = text_embeds.size(0)
    labels = torch.arange(batch_size).to(text_embeds.device)

    # Cross entropy loss
    loss_t2i = nn.functional.cross_entropy(logits_per_text, labels)
    loss_i2t = nn.functional.cross_entropy(logits_per_image, labels)

    return (loss_t2i + loss_i2t) / 2

def display_image(image):
    """
    Displays a batch of images (first sample only) in grayscale.
    Assumes `image` is shaped like (batch, height, width, 1).
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(image[0, :, :, 0], cmap="gray")
    plt.title("Reconstructed Image")
    plt.axis("off")
    plt.show()



def _to_numpy(img):
    """
    Accepts a PyTorch tensor, PIL.Image, or ndarray and returns a float32
    ndarray ∈ [0, 1] with shape (H, W).
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        if img.ndim == 4:            # (B,C,H,W) → take first B,C
            img = img[0, 0]
        elif img.ndim == 3:          # (C,H,W)
            img = img[0]
    elif isinstance(img, Image.Image):
        img = np.asarray(img)
    img = img.squeeze().astype(np.float32)
    if img.max() > 1.0:              # assume 8-bit
        img /= 255.0
    return img


def _histogram(arr, bins=256):
    """
    Returns a probability histogram (shape = (bins,)) for an image array
    assumed to be in [0, 1].
    """
    h, _ = np.histogram(arr, bins=bins, range=(0.0, 1.0), density=False)
    h = h.astype(np.float64)
    h /= h.sum() + 1e-12             # add ε to avoid division by zero
    return h


# -------------------------- distribution metrics -----------------------------
def _js_divergence(p, q, base=2):
    """
    Jensen–Shannon divergence (bounded 0…1 when base=2).
    """
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m, base=base) + 0.5 * entropy(q, m, base=base)


def _overlap_index(p, q):
    """
    Histogram overlap index = Σ min(pᵢ, qᵢ)  (a.k.a. intersection coefficient).
    Range 0…1 (1 = identical distributions).
    """
    return float(np.minimum(p, q).sum())


def compute_image_metrics(decoded_img,
                          target_img,
                          *,
                          bins: int = 256):
    """
    Returns a dict with **EMD, JSD, and Overlap Index** for one image pair.

    decoded_img : torch.Tensor | np.ndarray | PIL.Image
    target_img  : torch.Tensor | np.ndarray | PIL.Image
    """
    dec = _to_numpy(decoded_img)
    gt  = _to_numpy(target_img)

    # 1. Earth-Mover’s Distance (1-D)
    emd_val = wasserstein_distance(gt.flatten(), dec.flatten())

    # 2. Histogram-based metrics
    p = _histogram(gt, bins=bins)
    q = _histogram(dec, bins=bins)

    jsd_val = _js_divergence(p, q)          # 0 … 1   (lower is better)
    ovl_val = _overlap_index(p, q)          # 0 … 1   (higher is better)

    mse_val = float(np.mean((gt - dec) ** 2))
    psnr = float(10 * np.log10(1.0 / mse_val))  # Peak Signal-to-Noise Ratio

    return {
        "emd":  float(emd_val),
        "jsd":  float(jsd_val),
        "overlap": float(ovl_val),
        "mse": float(mse_val),
        "psnr": psnr,
    }


# --------------------------- JSON persistence --------------------------------
def _update_metrics_json(entry_key: str,
                         metrics_dict: dict,
                         json_path: str) -> None:
    """
    Append/overwrite metrics for `entry_key` in `json_path`.
    Creates parent dirs only if the file lives inside a directory.
    """
    dir_name = os.path.dirname(json_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    if os.path.isfile(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data[entry_key[24:-8]] = metrics_dict

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def save_metrics_for_image(decoded_img,
                           true_image_path: str,
                           json_metrics_path: str,
                           *,
                           bins: int = 256):
    """
    Convenience wrapper:  load ground-truth, compute metrics,
    and store them under the key  os.path.basename(true_image_path).
    """
    gt_img = Image.open(true_image_path).convert("L").resize((50, 50))

    metrics = compute_image_metrics(decoded_img, gt_img, bins=bins)
    _update_metrics_json(os.path.basename(true_image_path),
                         metrics,
                         json_metrics_path)
    return metrics

def pixels_to_slip(image,delta_z):

    """
    Converts pixel values in an image to slip values using a normalizing slip range.

    Input: image (numpy array) - The input image with pixel values.
           delta_z (float) - The scaling value for z axis used (Dz).

    Output: slip_valued_image (numpy array) - The image with pixel values converted to slip values.

    """

    normalizing_slip_range=np.load(r"./assets/normalizing_slip_range.npy",allow_pickle=True)
    slip_valued_image = (image *normalizing_slip_range)/delta_z

    return slip_valued_image