import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# File paths
scan_path = "//datasets/brats_2017/test_flair/Brats17_CBICA_AUN_1_flair.nii"
pred_mask_path = "//datasets/brats_2017/predictions_flair/Brats17_CBICA_AUN_1_flair_pred.nii"
gt_mask_path = "//datasets/brats_2017/WT/Brats17_CBICA_AUN_1_wt.nii"

# Load full 3D images
scan = nib.load(scan_path).get_fdata()
pred_mask = nib.load(pred_mask_path).get_fdata()
gt_mask = nib.load(gt_mask_path).get_fdata()

# Count zeros and ones across the full 3D volume
unique_pred, counts_pred = np.unique(pred_mask, return_counts=True)
unique_gt, counts_gt = np.unique(gt_mask, return_counts=True)

# Convert NumPy ints to standard Python ints and ensure {0,1} exist
pred_counts = {int(k): int(v) for k, v in zip(unique_pred, counts_pred)}
gt_counts = {int(k): int(v) for k, v in zip(unique_gt, counts_gt)}

# Ensure '1' appears in both dictionaries
pred_counts.setdefault(1, 0)
gt_counts.setdefault(1, 0)

# Print overall results
print(f"🔹 **Full 3D Scan Counts**")
print(f"✅ **Predicted Mask Counts**: {pred_counts}")
print(f"✅ **Ground Truth Mask Counts**: {gt_counts}")

# Plot middle slice as reference
slice_idx = 40 # Middle slice
scan_slice = scan[:, :, slice_idx]
pred_slice = pred_mask[:, :, slice_idx]
gt_slice = gt_mask[:, :, slice_idx]

# Plot images
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(scan_slice, cmap="gray")
axes[0].set_title(f"Original FLAIR Scan - Slice {slice_idx}")
axes[0].axis("off")

axes[1].imshow(gt_slice, cmap="gray")
axes[1].set_title(f"Ground Truth (WT) - Slice {slice_idx}")
axes[1].axis("off")

axes[2].imshow(pred_slice, cmap="gray")
axes[2].set_title(f"Predicted Mask - Slice {slice_idx}")
axes[2].axis("off")

axes[3].imshow(scan_slice, cmap="gray", alpha=0.5)
axes[3].imshow(pred_slice, cmap="jet", alpha=0.5)
axes[3].set_title("Overlay: Scan + Prediction")
axes[3].axis("off")

plt.tight_layout()
plt.show()