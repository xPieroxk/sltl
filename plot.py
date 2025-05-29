import sys
import os
from glob import glob
from triangulation import compute_areas, plot_tumor_evolution, plot_tumor_comparison

# Constants
DATASET_FOLDER = 'datasets/WT/'
MAX_PATIENTS = 210


def load_patient(index, patient_files):
    if index < 1 or index > len(patient_files):
        print(f"Error: Patient index {index} out of range. Must be between 1 and {len(patient_files)}.")
        sys.exit(1)
    path = patient_files[index - 1]  # 1-based indexing
    print(f"Loading patient #{index}: {path}")
    return path


def main():
    args = sys.argv[1:]
    if len(args) not in [1, 2]:
        print("Usage: python script.py <index1> [<index2>]")
        sys.exit(1)

    # Load patient file paths
    patient_files = sorted(glob(os.path.join(DATASET_FOLDER, '*.nii')))

    # Validate and convert arguments
    indices = []
    for arg in args:
        try:
            idx = int(arg)
            if idx < 1 or idx > MAX_PATIENTS:
                raise ValueError
            indices.append(idx)
        except ValueError:
            print(f"Invalid index '{arg}': Must be an integer between 1 and {MAX_PATIENTS}")
            sys.exit(1)

    if len(indices) == 1:
        p1 = load_patient(indices[0], patient_files)
        areas_1, _, _, _ = compute_areas(p1)
        patient_name = os.path.splitext(os.path.basename(p1))[0]
        plot_tumor_evolution(range(len(areas_1)), areas_1, title=f'Patient: {patient_name}')
    else:
        p1 = load_patient(indices[0], patient_files)
        p2 = load_patient(indices[1], patient_files)

        name1 = os.path.splitext(os.path.basename(p1))[0]
        name2 = os.path.splitext(os.path.basename(p2))[0]

        areas_1, _, _, _ = compute_areas(p1)
        areas_2, _, _, _ = compute_areas(p2)

        plot_tumor_comparison(
            time1=range(len(areas_1)),
            tumor_area1=areas_1,
            time2=range(len(areas_2)),
            tumor_area2=areas_2,
            title=f"{name1}(1) vs {name2}(2)"
        )


if __name__ == "__main__":
    main()
