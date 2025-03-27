import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from utils import convert_rgb_to_y
from tqdm import tqdm

def train(args):

    h5_file = h5py.File(args.output_path, 'w') #Opens an HDF5 file to store the preprocessed images.
    lr_patches = []
    hr_patches = []
    
    image_paths = sorted(glob.glob('{}/*'.format(args.images_dir)))
    
    for image_path in tqdm(image_paths, desc="Processing Images", unit="img"):
        hr = pil_image.open(image_path).convert('RGB')
        """ `Finds all images in the input directory.
            `Sorts them to ensure consistency.
            `Opens each image and converts it to RGB format."""


        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale

        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        
        lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        
        hr = np.array(hr).astype(np.float32) #Converts PIL images to NumPy arrays.
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        #Extracts small patches (e.g., 32x32) for training. (stride = 14)
        for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):   
            for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                lr_patches.append(lr[i:i + args.patch_size, j:j + args.patch_size])
                hr_patches.append(hr[i:i + args.patch_size, j:j + args.patch_size])

    #Converts lists to NumPy arrays.
    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)
    
    #Saves LR and HR patches into an HDF5 dataset.
    #Use this when you already have the full dataset (lr_patches) ready to be saved.
    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)
    h5_file.close()
    print("Completed...")

def eval(args):
    h5_file = h5py.File(args.output_path, 'w')

    # Use this when you want to organize multiple datasets into a structured hierarchy.
    lr_group = h5_file.create_group('lr') 
    hr_group = h5_file.create_group('hr')
    
    image_paths = sorted(glob.glob('{}/*'.format(args.images_dir)))

    for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths), desc="Processing Images", unit="img"):
        hr = pil_image.open(image_path).convert('RGB')
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        lr_group.create_dataset(str(i), data=lr) #Saves the full-size LR and HR images (no patches).
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()
    print("Completed...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--patch-size', type=int, default=32)
    parser.add_argument('--stride', type=int, default=14)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--eval', action='store_true')  
    args = parser.parse_args()

    #If --eval is not provided, it runs train().
    #If --eval is provided, it runs eval().
    if not args.eval:
        train(args)
    else:
        eval(args)