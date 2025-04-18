import numpy as np
import matplotlib.pyplot as plt
from skfuzzy import cmeans
from scipy.ndimage import binary_opening, binary_fill_holes, label, binary_dilation
from skimage.morphology import binary_closing

from utils import load_normalized_img, load_img, RANGES
from joblib import Parallel, delayed

def visualize_results(flair_slice, wt_slice, slice_idx, segmentation, segmented_image, sorted_clusters):
    fig, axes = plt.subplots(3, 6, figsize=(20, 12))
    axes = axes.flatten()

    axes[0].imshow(flair_slice, cmap="gray")
    axes[0].set_title(f"FLAIR Slice {slice_idx}")
    axes[0].axis("off")

    axes[1].imshow(wt_slice, cmap="gray")
    axes[1].set_title("Tumor Mask Slice")
    axes[1].axis("off")

    axes[2].imshow(segmentation, cmap="gray")
    axes[2].set_title("Final Tumor Segmentation")
    axes[2].axis("off")

    for i, (cluster_idx, intensity, size) in enumerate(sorted_clusters):
        cluster_mask = (segmented_image == cluster_idx).astype(np.uint8)
        axes[i + 3].imshow(cluster_mask, cmap="gray")
        axes[i + 3].set_title(f"Cluster {cluster_idx}")
        axes[i + 3].axis("off")

    plt.show()

def fuzzy_cmeans_clustering(image, num_clusters=15, m=2.0, error=1e-5, max_iter=1000):
    X = image.reshape(1, -1)
    cntr, U, _, _, _, _, _ = cmeans(X, num_clusters, m, error, max_iter, seed=42)
    return np.argmax(U, axis=0).reshape(image.shape), cntr


def get_sorted_clusters_by_intensity(image, segmented_image):
    cluster_intensities = []

    for cluster_idx in np.unique(segmented_image):
        cluster_mask = (segmented_image == cluster_idx)
        mean_intensity = np.mean(image[cluster_mask]) if np.sum(cluster_mask) > 0 else 0
        cluster_size = np.sum(cluster_mask)
        cluster_intensities.append((cluster_idx, mean_intensity, cluster_size))
    return sorted(cluster_intensities, key=lambda x: x[1], reverse=True)


def select_clusters(segmented_image, sorted_clusters, slice_idx, slices_size, start_index):
    norm_slice_idx = slice_idx - start_index
    fraction = norm_slice_idx / slices_size

    if fraction < 0.05 or fraction >= 0.95:
        top_n = 2
    elif fraction < 0.20 or fraction >= 0.80:
        top_n = 3
    elif fraction < 0.30 or fraction >= 0.70:
        top_n = 4
    elif fraction < 0.40 or fraction >= 0.60:
        top_n = 5
    else:
        top_n = 7

    print(f"Slice {slice_idx} (normalized={norm_slice_idx}, fraction={fraction:.2f}) -> top_n: {top_n}")

    top_clusters = [c[0] for c in sorted_clusters[:top_n]]
    return np.isin(segmented_image, top_clusters).astype(np.uint8)


def refine_segmentation(segmentation):
    segmentation = binary_opening(segmentation, structure=np.ones((3, 3)))
    largest_segmentation = binary_dilation(segmentation, structure=np.ones((3, 3)))
    largest_segmentation = binary_fill_holes(largest_segmentation)

    return largest_segmentation


def compute_metrics(gt_mask, pred_mask):
    TP = np.sum((gt_mask == 1) & (pred_mask == 1))
    FP = np.sum((gt_mask == 0) & (pred_mask == 1))
    FN = np.sum((gt_mask == 1) & (pred_mask == 0))
    TN = np.sum((gt_mask == 0) & (pred_mask == 0))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    return precision, sensitivity, specificity, f1_score, accuracy


def segment_tumor(flair_slice, wt_slice, slice_idx, slice_size, start_idx):
    # apply Fuzzy C-Means clustering
    segmented_image, _ = fuzzy_cmeans_clustering(flair_slice)

    # sort clusters by intensity
    sorted_clusters = get_sorted_clusters_by_intensity(flair_slice, segmented_image)

    # create initial tumor segmentation
    tumor_segmentation = select_clusters(segmented_image, sorted_clusters, slice_idx, slice_size, start_idx)

    # refine segmentation
    #tumor_segmentation = refine_segmentation(tumor_segmentation)

    # visualize
    visualize_results(flair_slice, wt_slice, slice_idx, tumor_segmentation, segmented_image, sorted_clusters)

    return tumor_segmentation


def process_selected_slices(flair_path, wt_path, selected_slices):
    # Load full 3D scans
    flair_img = load_normalized_img(flair_path)
    wt_mask = load_img(wt_path)

    segmentations = []
    ground_truths = []

    # Process only selected slices
    for slice_idx in selected_slices:
        flair_slice = flair_img[:, :, slice_idx]
        wt_slice = wt_mask[:, :, slice_idx]

        # Perform tumor segmentation
        tumor_segmentation = segment_tumor(flair_slice, wt_slice, slice_idx, len(selected_slices))

        # Store results for later evaluation
        segmentations.append(tumor_segmentation)
        ground_truths.append((wt_slice > 0).astype(np.uint8))  # Convert GT to binary

    # Stack all segmentations for evaluation
    segmentations = np.stack(segmentations, axis=-1)
    ground_truths = np.stack(ground_truths, axis=-1)

    # Compute metrics
    precision, recall, specificity, f1_score, accuracy = compute_metrics(ground_truths, segmentations)

    # Print final evaluation results
    print(f"Final Evaluation for Selected Slices:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    return segmentations, ground_truths


def process_selected_slices_parallel(flair_path, wt_path, selected_slices, start_idx, num_jobs=-1):
    # Load full 3D scans
    flair_img = load_normalized_img(flair_path)
    wt_mask = load_img(wt_path)

    # Parallel processing for segmentation
    results = Parallel(n_jobs=num_jobs)(
        delayed(segment_tumor)(flair_img[:, :, i], wt_mask[:, :, i], i, len(selected_slices), start_idx) for i in
        selected_slices
    )

    # Separate segmentations and ground truths
    segmentations = np.stack([r for r in results], axis=-1)
    ground_truths = np.stack([(wt_mask[:, :, i] > 0).astype(np.uint8) for i in selected_slices], axis=-1)

    # Compute metrics
    precision, recall, sensitivity, f1_score, accuracy = compute_metrics(ground_truths, segmentations)

    # Print final evaluation results
    print(f"Final Evaluation for Selected Slices:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    return segmentations, ground_truths

