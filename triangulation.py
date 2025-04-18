import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import Delaunay
import nibabel as nib


def compute_simplex_area(pts):
    '''
    Compute the area of a simplex given a list of points.
    '''
    A, B, C = pts
    return 0.5 * abs(A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))


def load_img(path):
    '''
    Load an image as float32
    '''
    return nib.load(path).get_fdata().astype(np.float32)


def get_tumor_contours(mask):
    '''
    Get tumor contours from a mask
    '''
    ctrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return ctrs


def get_image_corners(w, h):
    '''
    Get image corners
    '''
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.int32)


def filter_bg_simpleces(triangulation, points, mask, tumor_area):
    '''
    Filter bg simpleces by removing those are that are not part of the background.
    '''
    tumor_mask = (mask == 1)
    h, w = mask.shape
    valid_simplices = []
    total_area = (240 * 240) - tumor_area
    for simplex in triangulation.simplices:
        tri_pts = points[simplex]
        cx = np.mean(tri_pts[:, 0])
        cy = np.mean(tri_pts[:, 1])
        cx_i, cy_i = int(round(cx)), int(round(cy))
        if 0 <= cy_i < h and 0 <= cx_i < w:
            if not tumor_mask[cy_i, cx_i]:
                valid_simplices.append(simplex)
    return valid_simplices, total_area


def tumor_triangulation(tumor_contours, mask):
    '''
    Applies Delaunay triangulation algorithm to tumor region
    '''
    tumor_area = 0.0
    tumor_triangulations = []

    # iterates over each tumor region
    for c in tumor_contours:
        if len(c) > 2:  # at least 3 points are needed to define a closed shape
            pts = c.squeeze()
            if pts.ndim == 2 and pts.shape[0] >= 3:
                tri = Delaunay(pts)

                # Filter triangles to keep only those fully inside the tumor
                valid_simplices = []
                for simplex in tri.simplices:
                    tri_pts = pts[simplex]
                    cx, cy = np.mean(tri_pts, axis=0)  # Compute centroid
                    cx, cy = int(round(cx)), int(round(cy))

                    # Ensure centroid is inside the tumor mask
                    if 0 <= cx < mask.shape[1] and 0 <= cy < mask.shape[0]:
                        if mask[cy, cx] == 1:  # Check if inside tumor
                            valid_simplices.append(simplex)
                            tumor_area += compute_simplex_area(tri_pts)

                tumor_triangulations.append((pts, np.array(valid_simplices)))

    # create the tumor map
    tumor_matrix = np.zeros_like(mask, dtype=np.float32)
    tumor_matrix[mask == 1] = tumor_area
    return tumor_area, tumor_triangulations, tumor_matrix


def bg_triangulation(mask, tumor_contours, tumor_area):
    '''
    Applies Delaunay triangulation algorithm to background region
    '''
    h, w = mask.shape
    boundary_pts = get_image_corners(w, h)
    all_tumor_pts = []
    for c in tumor_contours:
        if len(c) > 2:
            p = c.squeeze()
            if p.ndim == 2 and p.shape[0] > 2:
                all_tumor_pts.append(p)
    if all_tumor_pts:
        merged_tumor_pts = np.vstack(all_tumor_pts)
        bg_points = np.vstack([boundary_pts, merged_tumor_pts])
    else:
        bg_points = boundary_pts
    tri_bg = Delaunay(bg_points)
    valid_simplices, bg_area = filter_bg_simpleces(tri_bg, bg_points, mask, tumor_area)

    # create the bg map
    bg_matrix = np.zeros_like(mask, dtype=np.float32)
    bg_matrix[mask == 0] = bg_area
    return bg_area, valid_simplices, bg_matrix


def plot_tumor_triangulation(mask, tumor_triangulations, point_size=4):
    '''
    Plot tumor triangulations
    '''
    # Create high-resolution figure for sharper visualization
    fig = plt.figure(figsize=(6, 6), dpi=200)  # Increase dpi here (try 200–300 for sharpness)
    plt.imshow(mask, cmap='gray', alpha=0.4)

    for pts, simplices in tumor_triangulations:
        plt.triplot(pts[:, 0], pts[:, 1], simplices, color='crimson', linewidth=0.6, alpha=0.9)
        plt.scatter(pts[:, 0], pts[:, 1], color='dodgerblue', s=point_size, alpha=0.8, edgecolors='none')

    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.title("Simplicial Complex Representation", fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.show()


def compute_areas(path):
    '''
    Compute areas of tumor and background regions
    '''
    # load img
    img = load_img(path)
    _, _, D = img.shape
    tumor_areas = []
    bg_areas = []
    bg_matrix = []
    tumor_matrix = []
    # iterate over slices
    for i in range(D):
        slice = img[:, :, i].astype(np.uint8)
        tumor_contours = get_tumor_contours(slice)
        # triangulation of tumor and background
        tumor_area, _, tumor_m = tumor_triangulation(tumor_contours, slice)
        bg_area, _, bg_m = bg_triangulation(slice, tumor_contours, tumor_area)
        # store values
        tumor_areas.append(tumor_area)
        tumor_matrix.append(tumor_m)
        bg_areas.append(bg_area)
        bg_matrix.append(bg_m)

    return tumor_areas, bg_areas, tumor_matrix, bg_matrix


def plot_background_triangulation(mask, bg_points, bg_simplices):
    '''
    Plot background triangulation
    '''
    plt.figure(figsize=(6, 6))
    plt.imshow((mask == 0).astype(np.uint8), cmap='gray', alpha=0.5)
    plt.triplot(bg_points[:, 0], bg_points[:, 1], bg_simplices, color='red')
    plt.scatter(bg_points[:, 0], bg_points[:, 1], color='blue', s=10)
    plt.xlim(0, mask.shape[1])
    plt.ylim(mask.shape[0], 0)
    plt.gca().set_aspect('equal')
    plt.title("Background Triangulation")
    plt.show()


def plot_tumor_triangulation(mask, tumor_triangulations):
    '''
    Plot tumor triangulation
    '''
    plt.figure(figsize=(8, 8), dpi=100)

    # Show the background mask with slight transparency
    plt.imshow(mask, cmap='gray', alpha=0.4)

    # Plot triangulations and points
    for pts, simplices in tumor_triangulations:
        plt.triplot(pts[:, 0], pts[:, 1], simplices, color='#e74c3c', linewidth=1.2, alpha=0.9)
        plt.scatter(pts[:, 0], pts[:, 1], color='#2980b9', s=15, edgecolor='white', linewidth=0.5)

    # Clean up the axes
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_aspect('equal')

    # Set a more elegant title
    plt.title("Simplicial Complex Representation of Tumor Region", fontsize=14, fontweight='semibold', pad=20)

    # Optional: add a border or subtle background grid if desired
    # ax.set_facecolor('#f9f9f9')  # light background, if needed

    plt.tight_layout()
    plt.show()


def plot_tumor_comparison(time1, tumor_area1, time2, tumor_area2, label1="Case 1", label2="Case 2"):
    '''
    Plot tumor area comparison between 2 areas
    '''
    plt.figure(figsize=(8, 5))

    # Plot first case
    plt.plot(time1, tumor_area1, linestyle='-', color='red', linewidth=2, label=label1)

    # Plot second case
    plt.plot(time2, tumor_area2, linestyle='-', color='blue', linewidth=2, label=label2)

    # Labels and title
    plt.xlabel("Slices", fontsize=12, fontweight='bold')
    plt.ylabel("Tumor Area", fontsize=12, fontweight='bold')
    plt.title("Comparison of Tumor Area Evolution", fontsize=14, fontweight='bold')

    # Improve visualization
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend()

    plt.show()
