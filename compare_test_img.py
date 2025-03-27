import argparse
import os
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure

# Define transformation (Convert OpenCV images to PyTorch tensors)
transform = transforms.Compose([
    transforms.ToTensor()  # Converts (H, W, C) OpenCV images to (C, H, W) PyTorch tensors
])

# Function to resize image to match another image's dimensions
def resize_to_match(image, target_size):
    return cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_CUBIC)

# Function to calculate PSNR, MSE, and SSIM between two images
def compare_images(source_image, target_image):
    mse_value = F.mse_loss(source_image, target_image).item()  # Compute MSE
    psnr_value = 20 * torch.log10(1.0 / torch.sqrt(F.mse_loss(source_image, target_image))).item()  # Compute PSNR
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)  # Initialize SSIM metric
    ssim_value = ssim_metric(source_image.unsqueeze(0), target_image.unsqueeze(0)).item()  # Compute SSIM

    return psnr_value, mse_value, ssim_value

# Function to compare a source image with multiple target images
def compare_source_with_multiple(source_path, target_dir, max_images):
    # Ensure source image exists
    if not os.path.exists(source_path):
        print(f"Error: Source image '{source_path}' not found.")
        return []

    # Load source image
    source_image = cv2.imread(source_path)
    if source_image is None:
        print(f"Error: Could not load source image '{source_path}'.")
        return []

    source_size = source_image.shape[:2]  # Get source image size (height, width)
    source_image = transform(source_image)  # Convert to tensor

    count = 0
    results = []

    for file in os.listdir(target_dir):
        if count >= max_images:
            break

        target_path = os.path.join(target_dir, file)

        # Ensure target image exists
        if not os.path.exists(target_path):
            print(f"Skipping {file}: Image is missing.")
            continue

        # Load target image
        target_image = cv2.imread(target_path)
        if target_image is None:
            print(f"Skipping {file}: Could not load image.")
            continue

        try:
            # Resize target image to match the source image size
            target_image = resize_to_match(target_image, source_size)
            
            # Convert to tensor
            target_image = transform(target_image)

            # Compute image quality metrics
            psnr, mse, ssim = compare_images(source_image, target_image)

            # Store results
            results.append({"Filename": file, "PSNR": psnr, "MSE": mse, "SSIM": ssim})

            # Print results
            print(f"Comparing Source with: {file}")
            print(f"PSNR: {psnr:.2f} dB")
            print(f"MSE: {mse:.8f}")
            print(f"SSIM: {ssim:.4f}\n")

            count += 1
        except Exception as e:
            print(f"Error processing {file}: {e}")

    return results  # Return results for further analysis

# Main function with argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare a source image with multiple images using PSNR, MSE, and SSIM.")
    parser.add_argument("--source", type=str, required=True, help="Path to the source image.")
    parser.add_argument("--target-dir", type=str, required=True, help="Path to the directory containing target images.")
    parser.add_argument("--max-images", type=int, default=5, help="Maximum number of images to compare.")
    
    args = parser.parse_args()

    # Run comparison
    compare_source_with_multiple(args.source, args.target_dir, args.max_images)
