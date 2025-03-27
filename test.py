import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr


if __name__ == '__main__':
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Test SRCNN on an image.")
    parser.add_argument('--weights-file', type=str, required=True, help="Path to the trained SRCNN model weights (.pth).")
    parser.add_argument('--image-file', type=str, required=True, help="Path to the input HR image.")
    parser.add_argument('--scale', type=int, default=3, help="Upscaling factor (e.g., 2, 3, 4).")
    args = parser.parse_args()

    # Enable CUDA benchmark mode for faster inference
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the SRCNN model
    model = SRCNN().to(device)
    state_dict = model.state_dict()

    # Load trained weights
    for n, p in torch.load(args.weights_file, map_location=device).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(f"Unexpected key {n} in weights file.")

    model.eval()  # Switch to evaluation mode

    # Load image and convert to RGB
    image = pil_image.open(args.image_file).convert('RGB')

    # Resize image to be divisible by scale factor
    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)

    # Generate low-resolution (LR) image via bicubic downsampling and upsampling
    lr_image = image.resize((image_width // args.scale, image_height // args.scale), resample=pil_image.BICUBIC)
    lr_image = lr_image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr_image.save(args.image_file.replace('.', f'_bicubic_x{args.scale}.'))

    # Convert to numpy array and extract Y channel
    image_np = np.array(lr_image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image_np)
    y_channel = ycbcr[..., 0] / 255.0  # Normalize Y channel

    # Convert Y channel to tensor
    y_tensor = torch.from_numpy(y_channel).to(device).unsqueeze(0).unsqueeze(0)

    # Run SRCNN on the Y channel
    with torch.no_grad():
        sr_y = model(y_tensor).clamp(0.0, 1.0)

    # Compute PSNR between the original and SRCNN output
    psnr = calc_psnr(y_tensor, sr_y)
    print('PSNR: {:.2f} dB'.format(psnr))

    # Convert SRCNN output back to an image
    sr_y_np = sr_y.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    output = np.array([sr_y_np, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)

    # Save the SRCNN output
    output.save(args.image_file.replace('.', f'_srcnn_x{args.scale}.'))
    
    
    
    
    
    
    
