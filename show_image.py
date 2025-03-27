import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from utils import calc_psnr

# Argument parser for input images and patch parameters
parser = argparse.ArgumentParser(description="Display three images with extracted patches and compute PSNR.")
parser.add_argument("--original", type=str, required=True, help="Path to the original image")
parser.add_argument("--bicubic", type=str, required=True, help="Path to the bicubic image")
parser.add_argument("--srcnn", type=str, required=True, help="Path to the SRCNN image")
parser.add_argument("--patch_x", type=int, default=50, help="X coordinate for patch (default: 50)")
parser.add_argument("--patch_y", type=int, default=50, help="Y coordinate for patch (default: 50)")
parser.add_argument("--patch_size", type=int, default=256, help="Patch size (default: 256x256)")
args = parser.parse_args()

# Load images
original = cv2.cvtColor(cv2.imread(args.original), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
bicubic = cv2.cvtColor(cv2.imread(args.bicubic), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
srcnn = cv2.cvtColor(cv2.imread(args.srcnn), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

# Convert images to PyTorch tensors for PSNR calculation
original_torch = torch.tensor(original).permute(2, 0, 1)  # Change shape to (C, H, W)
bicubic_torch = torch.tensor(bicubic).permute(2, 0, 1)
srcnn_torch = torch.tensor(srcnn).permute(2, 0, 1)

# Compute PSNR values
psnr_bicubic = calc_psnr(original_torch, bicubic_torch).item()
psnr_srcnn = calc_psnr(original_torch, srcnn_torch).item()

# Extract patches from each image
patch_size = args.patch_size
patch_x, patch_y = args.patch_x, args.patch_y

patch_original = original[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size]
patch_bicubic = bicubic[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size]
patch_srcnn = srcnn[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size]

# Draw a red rectangle on each image at the patch location
color = (1, 0, 0)  # Red color in normalized RGB
thickness = 3

cv2.rectangle(original, (patch_x, patch_y), (patch_x + patch_size, patch_y + patch_size), color, thickness)
cv2.rectangle(bicubic, (patch_x, patch_y), (patch_x + patch_size, patch_y + patch_size), color, thickness)
cv2.rectangle(srcnn, (patch_x, patch_y), (patch_x + patch_size, patch_y + patch_size), color, thickness)

# Create a figure (3 images in top row, 3 patches below)
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Display full images with red rectangles and PSNR values
axs[0, 0].imshow(original)
axs[0, 0].set_title("Original", fontsize=14, color="orange", fontweight="bold")
axs[0, 0].axis("off")

axs[0, 1].imshow(bicubic)
axs[0, 1].set_title(f"Bicubic (PSNR: {psnr_bicubic:.2f} dB)", fontsize=14, color="orange", fontweight="bold")
axs[0, 1].axis("off")

axs[0, 2].imshow(srcnn)
axs[0, 2].set_title(f"SRCNN (PSNR: {psnr_srcnn:.2f} dB)", fontsize=14, color="orange", fontweight="bold")
axs[0, 2].axis("off")

# Display patches below their respective images
axs[1, 0].imshow(patch_original)
axs[1, 0].set_title("Patch from Original", fontsize=12, color="cyan", fontweight="bold")
axs[1, 0].axis("off")

axs[1, 1].imshow(patch_bicubic)
axs[1, 1].set_title("Patch from Bicubic", fontsize=12, color="cyan", fontweight="bold")
axs[1, 1].axis("off")

axs[1, 2].imshow(patch_srcnn)
axs[1, 2].set_title("Patch from SRCNN", fontsize=12, color="cyan", fontweight="bold")
axs[1, 2].axis("off")

# Adjust layout and show
plt.tight_layout()
plt.show()
