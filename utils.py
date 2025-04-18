import os
import nibabel as nib
import numpy as np
import shutil
import pandas as pd

DATASET_FOLDER = os.path.join("datasets", "brats_2017")
HGG_FOLDER = os.path.join(DATASET_FOLDER, "Brats17TrainingData", "HGG")
WT_FOLDER = os.path.join(DATASET_FOLDER, "WT")
MULTIMODAL_FOLDER = os.path.join(DATASET_FOLDER, "multimodal")

TRAIN_FLAIR_FOLDER = os.path.join(DATASET_FOLDER, "train_flair")
TRAIN_FLAIR_LABELS_FOLDER = os.path.join(DATASET_FOLDER, "train_flair_labels")
TEST_FLAIR_FOLDER = os.path.join(DATASET_FOLDER, "test_flair")
TEST_FLAIR_LABELS_FOLDER = os.path.join(DATASET_FOLDER, "test_flair_labels")
PREDICTIONS_FLAIR_FOLDER = os.path.join(DATASET_FOLDER, "predictions_flair")

MODEL_FILE = os.path.join(DATASET_FOLDER, "tumor_classifier.keras")

CLUSTER_SIZE_SMALL = 110
CLUSTER_SIZE_LARGE = 170

ONE_CLUSTER_RANGE = [38840, 49000]
TWO_CLUSTER_RANGE = [38900, 48000]
THREE_CLUSTER_RANGE = [40000, 46000]
FOUR_CLUSTER_RANGE = [42000, 44000]

RANGES = [
    (ONE_CLUSTER_RANGE, 1),
    (TWO_CLUSTER_RANGE, 2),
    (THREE_CLUSTER_RANGE, 3),
    (FOUR_CLUSTER_RANGE, 4)
]


# 5 CLUSTER IN THE MIDDLE

def compute_whole_tumor_masks():
    patient_folders = [f for f in os.listdir(HGG_FOLDER) if os.path.isdir(os.path.join(HGG_FOLDER, f))]
    for patient_id in patient_folders:
        segmentation_file = os.path.join(HGG_FOLDER, patient_id, f"{patient_id}_seg.nii")

        # load segmentation
        segmentation_image = nib.load(segmentation_file)
        segmentation_data = segmentation_image.get_fdata()

        # WT labels 1, 2, 4
        whole_tumor_mask = (segmentation_data > 0).astype(np.uint8)

        # save WT mask
        wt_output_path = os.path.join(WT_FOLDER, f"{patient_id}_wt.nii")
        wt_nifti = nib.Nifti1Image(whole_tumor_mask, affine=segmentation_image.affine)
        nib.save(wt_nifti, wt_output_path)
        print(f'{wt_output_path} saved')


def compute_multimodal_data():
    modalities = ["t1", "t1ce", "t2", "flair"]
    patient_folders = [f for f in os.listdir(HGG_FOLDER) if os.path.isdir(os.path.join(HGG_FOLDER, f))]

    for patient_id in patient_folders:
        patient_folder = os.path.join(HGG_FOLDER, patient_id)

        # load modalities
        multi_modal = []
        for modality in modalities:
            file_path = os.path.join(patient_folder, f"{patient_id}_{modality}.nii")
            multi_modal.append(nib.load(file_path).get_fdata())

        # stack and save modalities
        multi_modal_array = np.stack(multi_modal, axis=-1)
        output_path = os.path.join(MULTIMODAL_FOLDER, f"{patient_id}.npz")
        np.savez_compressed(output_path, multi_modal=multi_modal_array)
        print(f"{output_path} saved")


def copy_single_modality_data(modality="flair"):
    patient_folders = [f for f in os.listdir(HGG_FOLDER) if os.path.isdir(os.path.join(HGG_FOLDER, f))]

    for patient_id in patient_folders:
        patient_folder = os.path.join(HGG_FOLDER, patient_id)
        source_path = os.path.join(patient_folder, f"{patient_id}_{modality}.nii")
        destination_path = os.path.join(DATASET_FOLDER, f"training_{modality}", f"{patient_id}_{modality}.nii")
        # copy the modality
        shutil.copy(source_path, destination_path)


def get_affine_matrix(patient_id):
    patient_folder = patient_id.rsplit("_", 1)[0]
    t1_file = os.path.join(HGG_FOLDER, patient_folder, patient_id)
    return nib.load(t1_file).affine


def generate_slice_labels(flair_dir, wt_dir, output_dir):
    # List all FLAIR files
    flair_files = sorted([f for f in os.listdir(flair_dir) if f.endswith(".nii")])

    for flair_file in flair_files:
        # Construct matching WT file
        wt_file = flair_file.replace("_flair.nii", "_wt.nii")

        # Load NIfTI images
        flair_path = os.path.join(flair_dir, flair_file)
        wt_path = os.path.join(wt_dir, wt_file)

        flair_img = nib.load(flair_path).get_fdata()
        wt_mask = nib.load(wt_path).get_fdata()

        # Storage for current patient dataset
        patient_data = []

        # Iterate over slices
        for slice_idx in range(flair_img.shape[2]):
            wt_slice = wt_mask[:, :, slice_idx]
            contains_tumor = int(np.any(wt_slice > 0))
            patient_data.append([flair_file, slice_idx, contains_tumor])

        # save data
        output_csv_path = os.path.join(output_dir, f"{flair_file}_labels.csv")

        df = pd.DataFrame(patient_data, columns=["filename", "slice_idx", "label"])
        df.to_csv(output_csv_path, index=False)

        print(f"Saved: {output_csv_path}")


def load_normalized_img(path):
    img = nib.load(path).get_fdata()
    return (img - np.mean(img)) / (np.std(img) + 1e-8)


def load_img(path):
    return nib.load(path).get_fdata()
