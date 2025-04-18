import numpy as np
import nibabel as nib

# Define file paths
flair_path = "//datasets/brats_2017/test_flair/Brats17_CBICA_AUN_1_flair.nii"
wt_path = "//datasets/brats_2017/WT/Brats17_CBICA_AUN_1_wt.nii"

# Load NIfTI images
flair_img = nib.load(flair_path).get_fdata()
wt_mask = nib.load(wt_path).get_fdata()  # Binary mask (0: background, 1: tumor)

# Normalize FLAIR Image to [0,1] range
flair_img = (flair_img - np.min(flair_img)) / (np.max(flair_img) - np.min(flair_img))

# Get number of slices (depth)
num_slices = wt_mask.shape[2]
print("Per Slice Tumor Pixel Count:")
print("=" * 50)

# Iterate over slices
for slice_idx in range(num_slices):
    tumor_pixel_count = np.sum(wt_mask[:, :, slice_idx] > 0)  # Count nonzero (tumor) pixels

    if tumor_pixel_count > 0:  # Only print if there are tumor pixels in this slice
        print(f"Slice {slice_idx}: Tumor Area = {tumor_pixel_count} pixels")

# Print summary
print("\nTotal Slices with Tumor:", np.sum(np.sum(wt_mask, axis=(0, 1)) > 0))