import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np

def view_h5_dataset(h5_path):
    """
    Loads and visualizes HR and LR image patches from an HDF5 file.
    Supports both direct datasets and groups inside the file.
    """

    with h5py.File(h5_path, "r") as h5_file:
        print("Datasets & Groups inside the HDF5 file:", list(h5_file.keys()))

        def get_first_dataset(group):
            """Retrieve the first dataset inside a group."""
            first_key = list(group.keys())[0]  # Get first dataset name
            return np.array(group[first_key]), first_key  # Return dataset & key

        # Determine if 'hr' and 'lr' are groups or datasets
        if isinstance(h5_file["hr"], h5py.Group):  
            hr_patch, hr_key = get_first_dataset(h5_file["hr"])  # Fetch first dataset inside HR group
            lr_patch, lr_key = get_first_dataset(h5_file["lr"])  # Fetch first dataset inside LR group
            print(f"HR Patch Shape ({hr_key}): {hr_patch.shape}")
            print(f"LR Patch Shape ({lr_key}): {lr_patch.shape}")
        else:
            hr_patch = np.array(h5_file["hr"][0])  # Fetch first HR dataset directly
            lr_patch = np.array(h5_file["lr"][0])  # Fetch first LR dataset directly
            print(f"HR Patch Shape: {hr_patch.shape}")
            print(f"LR Patch Shape: {lr_patch.shape}")

        # Display patches
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(lr_patch, cmap="gray")
        axs[0].set_title("Low-Resolution Patch")
        axs[1].imshow(hr_patch, cmap="gray")
        axs[1].set_title("High-Resolution Patch")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View HDF5 dataset contents.")
    parser.add_argument("--h5_file", type=str, required=True, help="Path to the HDF5 file")

    args = parser.parse_args()
    view_h5_dataset(args.h5_file)
