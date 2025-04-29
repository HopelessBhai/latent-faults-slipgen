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