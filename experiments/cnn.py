import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras import layers, models, losses, metrics
from tensorflow.keras.applications import ResNet50
import random
from utils import WT_FOLDER, TEST_FLAIR_FOLDER, PREDICTIONS_FLAIR_FOLDER, \
    get_affine_matrix, TRAIN_FLAIR_FOLDER


def UNet2D_multimodal(input_shape=(240, 240, 4)):
    inputs = layers.Input(input_shape)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(x)
    return models.Model(inputs, outputs)


def UNet2D_single_modality(input_shape=(240, 240, 1)):
    inputs = layers.Input(input_shape)

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)

    x = layers.UpSampling2D((2, 2))(x)
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(x)

    return models.Model(inputs, outputs)


def UNet_ResNet50(input_shape=(240, 240, 1), trainable_layers=20):
    # Load ResNet50 as encoder
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(input_shape[0], input_shape[1], 3))

    # Freeze lower layers of ResNet, train only the last `trainable_layers`
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False

    # Image Input (Grayscale to 3 channels)
    flair_input = layers.Input(shape=input_shape, name="flair_input")
    x = layers.Conv2D(3, (1, 1), activation="relu")(flair_input)  # Convert grayscale to RGB
    x = base_model(x)

    # Decoder (Upsampling Path) - Ensures output is (240, 240)
    x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)  # 8 → 16
    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)  # 16 → 32
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)  # 32 → 64
    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)  # 64 → 128
    x = layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same", activation="relu")(x)  # 128 → 256

    # **Fixing Size Mismatch**
    x = layers.Conv2DTranspose(8, (3, 3), strides=(1, 1), padding="same", activation="relu")(x)  # Ensure finer control
    x = layers.Cropping2D(cropping=((8, 8), (8, 8)))(x)  # Crop from 256 → 240

    # Output Layer (Binary Segmentation)
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid", padding="same")(x)

    # Build & Compile Model
    model = models.Model(inputs=flair_input, outputs=outputs)
    focal_loss = tf.keras.losses.BinaryFocalCrossentropy(alpha=0.25, gamma=2)
    model.compile(optimizer="adam", loss=focal_loss, metrics=[
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5, name="iou")
    ])

    return model


def UNet3D_single_modality(input_shape=(240, 240, 155, 1)):  # Fixed input shape
    inputs = layers.Input(input_shape)

    # Encoder
    x = layers.Conv3D(32, (3, 3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling3D((2, 2, 2), padding="same")(x)

    x = layers.Conv3D(64, (3, 3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling3D((2, 2, 2), padding="same")(x)

    x = layers.Conv3D(128, (3, 3, 3), activation="relu", padding="same")(x)

    # Decoder
    x = layers.UpSampling3D((2, 2, 2))(x)  # Normal integer scaling
    x = layers.Conv3D(64, (3, 3, 3), activation="relu", padding="same")(x)

    x = layers.UpSampling3D((2, 2, 2))(x)  # Normal integer scaling
    x = layers.Conv3D(1, (1, 1, 1), activation="sigmoid", padding="same")(x)

    # Crop depth to 155
    x = layers.Cropping3D(((0, 0), (0, 0), (0, x.shape[3] - 155)))(x)  # Crop extra slices

    return models.Model(inputs, x)


def SE_Block(x, ratio=16):
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(filters // ratio, activation="relu")(se)
    se = layers.Dense(filters, activation="sigmoid")(se)
    return layers.Multiply()([x, se])


def ResBlock(x, filters):
    res = layers.Conv2D(filters, (1, 1), padding="same")(x)  # 1x1 conv for matching dims
    x = layers.Conv2D(filters, (3, 3), padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (3, 3), padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, res])  # Residual connection
    x = layers.ReLU()(x)
    return x


def AttentionGate(x, g):
    inter_channels = x.shape[-1] // 2
    theta_x = layers.Conv2D(inter_channels, (1, 1), padding="same")(x)
    phi_g = layers.Conv2D(inter_channels, (1, 1), padding="same")(g)
    attn = layers.ReLU()(layers.Add()([theta_x, phi_g]))
    attn = layers.Conv2D(1, (1, 1), activation="sigmoid")(attn)
    return layers.Multiply()([x, attn])


def AdvancedUNetBinary(input_shape=(240, 240, 1)):
    inputs = layers.Input(input_shape)

    # Encoder
    c1 = ResBlock(inputs, 32)
    c1 = SE_Block(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = ResBlock(p1, 64)
    c2 = SE_Block(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = ResBlock(p2, 128)
    c3 = SE_Block(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    b = ResBlock(p3, 256)
    b = SE_Block(b)

    # Decoder
    u1 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(b)
    u1 = AttentionGate(c3, u1)
    u1 = layers.Concatenate()([u1, c3])
    u1 = ResBlock(u1, 128)

    u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(u1)
    u2 = AttentionGate(c2, u2)
    u2 = layers.Concatenate()([u2, c2])
    u2 = ResBlock(u2, 64)

    u3 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(u2)
    u3 = AttentionGate(c1, u3)
    u3 = layers.Concatenate()([u3, c1])
    u3 = ResBlock(u3, 32)

    # Output Layer (Binary segmentation)
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(u3)

    return models.Model(inputs, outputs)


def load_patient_2d(patient_id, training_folder):
    nii_path = os.path.join(training_folder, f"{patient_id}.nii")
    mask_path = os.path.join(WT_FOLDER, f"{patient_id.rsplit('_', 1)[0]}_wt.nii")
    # load FLAIR modality
    data = nib.load(nii_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()

    # normalize image (0 to 1 range)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # ensure correct shape: (D, H, W, 1)
    mri_slices = np.expand_dims(np.transpose(data, (2, 0, 1)), axis=-1)
    mask_slices = np.expand_dims(np.transpose(mask, (2, 0, 1)), axis=-1)

    return mri_slices, mask_slices


def train_2d(model, training_folder, num_epochs=5, batch_size=16, save_path="unet_brats_first.keras"):
    patient_ids = [f.split('.')[0] for f in os.listdir(training_folder) if f.endswith(".nii")]
    patient_ids = patient_ids[:5]
    for epoch in range(num_epochs):
        # shuffle patients
        random.shuffle(patient_ids)

        for patient_id in patient_ids:
            print(f"Training on: {patient_id}")
            X_train, Y_train = load_patient_2d(patient_id, training_folder)
            class_weight = {0: 1.0, 1: 5.0}  # Higher weight for tumor pixels

            model.fit(
                X_train, Y_train,
                epochs=num_epochs,
                batch_size=batch_size,
                sample_weight=np.array([class_weight.get(int(y), 1.0) for y in Y_train.flatten()]).reshape(
                    Y_train.shape)
            )

    model.save(save_path)
    print(f"Model saved at: {save_path}")


def load_patient_3d(patient_id, training_folder):
    nii_path = os.path.join(training_folder, f"{patient_id}.nii")
    mask_path = os.path.join(WT_FOLDER, f"{patient_id.rsplit('_', 1)[0]}_wt.nii")

    # Load full 3D FLAIR scan
    data = nib.load(nii_path).get_fdata()  # Shape: (H, W, D)
    mask = nib.load(mask_path).get_fdata()  # Shape: (H, W, D)

    # Normalize the volume (0-1 range)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Keep original shape but add channel dimension
    mri_volume = np.expand_dims(data, axis=-1)  # Shape: (H, W, D, 1)
    mask_volume = np.expand_dims(mask, axis=-1)  # Shape: (H, W, D, 1)

    return mri_volume, mask_volume


def train_3d(model, training_folder, num_epochs=5, batch_size=2, save_path="unet_brats_3d.keras"):
    patient_ids = [f.split('.')[0] for f in os.listdir(training_folder) if f.endswith(".nii")]

    for epoch in range(num_epochs):
        random.shuffle(patient_ids)  # Shuffle patients
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for i in range(0, len(patient_ids[:4]), batch_size):  # Process patients in batches
            batch_ids = patient_ids[i:i + batch_size]

            X_train_batch, Y_train_batch = [], []
            for patient_id in batch_ids:
                print(f"Loading: {patient_id}")
                X_train, Y_train = load_patient_3d(patient_id, training_folder)  # Load 3D data
                X_train_batch.append(X_train)
                Y_train_batch.append(Y_train)

            # Convert to NumPy arrays for batch processing
            X_train_batch = np.array(X_train_batch)
            Y_train_batch = np.array(Y_train_batch)
            # Train on batch
            model.fit(X_train_batch, Y_train_batch, epochs=1, batch_size=1, verbose=1)

    model.save(save_path)
    print(f"Model saved at: {save_path}")


def predict_2d(model, test_folder, predictions_folder):
    patient_ids = [f.split('.')[0] for f in os.listdir(test_folder) if f.endswith(".nii")]

    for patient_id in patient_ids:
        print(f"Predicting for: {patient_id}")

        nii_path = os.path.join(test_folder, f"{patient_id}.nii")

        # load patient scan
        data = nib.load(nii_path).get_fdata()
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

        # ensure correct shape: (D, H, W, 1)
        test_slices = np.expand_dims(np.transpose(data, (2, 0, 1)), axis=-1)

        # run prediction
        predicted_mask = (model.predict(test_slices) > 0.5).astype(np.uint8)

        # remove extra channel (D, H, W, 1) → (D, H, W)
        predicted_mask = np.squeeze(predicted_mask)
        # convert to (H, W, D)
        predicted_mask = np.transpose(predicted_mask, (1, 2, 0))

        # save prediction
        affine = get_affine_matrix(f"{patient_id}.nii")
        output_path = os.path.join(predictions_folder, f"{patient_id}_pred.nii")
        nib.save(nib.Nifti1Image(predicted_mask, affine), output_path)
        print(f"{output_path} saved")


def predict_3d(model, test_folder, predictions_folder):
    patient_ids = [f.split('.')[0] for f in os.listdir(test_folder) if f.endswith(".nii")]

    for patient_id in patient_ids:
        print(f"Predicting for: {patient_id}")

        nii_path = os.path.join(test_folder, f"{patient_id}.nii")

        # Load 3D patient scan
        data = nib.load(nii_path).get_fdata()
        data = (data - np.min(data)) / (np.max(data) - np.min(data))  # Normalize intensity

        # Ensure correct shape: (H, W, D, 1) → suitable for Conv3D
        test_volume = np.expand_dims(data, axis=-1)  # Add channel dimension
        test_volume = np.expand_dims(test_volume, axis=0)  # Add batch dimension (1, H, W, D, 1)

        # Run 3D prediction
        predicted_mask = (model.predict(test_volume) > 0.5).astype(np.uint8)

        # Remove batch and channel dimensions: (1, H, W, D, 1) → (H, W, D)
        predicted_mask = np.squeeze(predicted_mask)

        # Save prediction as NIfTI file
        affine = nib.load(nii_path).affine  # Get affine matrix for correct spatial mapping
        output_path = os.path.join(predictions_folder, f"{patient_id}_pred.nii")
        nib.save(nib.Nifti1Image(predicted_mask, affine), output_path)

        print(f"{output_path} saved")


'''model_path = "/Users/pierohierro/Desktop/university/msc_thesis/work/code/sltl/unet_brats_first1.keras"
model = tf.keras.models.load_model(model_path)'''
model = UNet_ResNet50()
# Train model
train_2d(model, TRAIN_FLAIR_FOLDER)

# Predict on test set
predict_2d(model, TEST_FLAIR_FOLDER, PREDICTIONS_FLAIR_FOLDER)
