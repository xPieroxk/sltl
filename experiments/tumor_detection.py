import random
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, \
    Concatenate, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from utils import load_normalized_img, TRAIN_FLAIR_FOLDER, TRAIN_FLAIR_LABELS_FOLDER, TEST_FLAIR_FOLDER, \
    TEST_FLAIR_LABELS_FOLDER, MODEL_FILE


def build_2d_cnn(input_shape=(240, 240, 1)):
    # ---- FLAIR Image Input ----
    flair_input = Input(shape=input_shape, name="flair_input")

    # First Conv Block
    x = Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same")(flair_input)
    x = BatchNormalization()(x)
    x = Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Second Conv Block
    x = Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Third Conv Block
    x = Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten Features
    x = Flatten()(x)

    # ---- Slice Index Input ----
    slice_input = Input(shape=(1,), name="slice_index_input")  # Single scalar value

    # ---- Merge CNN Features & Slice Index ----
    merged = Concatenate()([x, slice_input])

    # Fully Connected Layers
    merged = Dense(256, activation="relu")(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(128, activation="relu")(merged)
    merged = Dropout(0.5)(merged)

    # Output Layer (Binary Classification)
    output = Dense(1, activation="sigmoid", name="output")(merged)

    # Build Model
    model = Model(inputs=[flair_input, slice_input], outputs=output)

    # Compile Model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


def build_resnet50_finetune(input_shape=(240, 240, 1)):
    # ---- Flair Image Input ----
    flair_input = Input(shape=input_shape, name="flair_input")

    # Convert grayscale (1-channel) → 3-channel using a **learnable** 1×1 convolution
    x = Conv2D(3, (1, 1), activation="relu", padding="same")(flair_input)

    # Load **ResNet50 (pretrained on ImageNet)**, removing the top layers
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(240, 240, 3))

    # Fine-Tuning: Unfreeze last few layers for training
    for layer in base_model.layers[-20:]:  # Unfreezing last 20 layers
        layer.trainable = True

    # Pass transformed grayscale MRI image through ResNet50
    x = base_model(x, training=True)  # Ensure it updates the fine-tuned layers
    x = GlobalAveragePooling2D()(x)  # Reduce spatial dimensions

    # ---- Slice Index Input ----
    slice_input = Input(shape=(1,), name="slice_index_input")

    # ---- Merge Features from ResNet50 & Slice Index ----
    merged = Concatenate()([x, slice_input])

    # Fully Connected Layers
    merged = Dense(256, activation="relu")(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(128, activation="relu")(merged)
    merged = Dropout(0.5)(merged)

    # Output Layer (Binary Classification)
    output = Dense(1, activation="sigmoid", name="output")(merged)

    # Build Final Model
    model = Model(inputs=[flair_input, slice_input], outputs=output)

    # Compile Model with a lower learning rate for fine-tuning
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    return model


def load_patient_slices(patient_id, flair_dir, labels_csv_dir):
    flair_path = os.path.join(flair_dir, f"{patient_id}_flair.nii")
    labels_csv_path = os.path.join(labels_csv_dir, f"{patient_id}_flair.nii_labels.csv")

    flair_img = load_normalized_img(flair_path)
    labels_df = pd.read_csv(labels_csv_path)

    X_slices = [flair_img[:, :, i] for i in range(flair_img.shape[2])]
    Y_slices = labels_df["label"].values

    slice_indices = np.arange(len(X_slices)) / (len(X_slices) - 1)

    X_slices = np.expand_dims(X_slices, axis=-1)
    slice_indices = np.expand_dims(slice_indices, axis=-1)

    return [np.array(X_slices), np.array(slice_indices)], np.array(Y_slices)


def load_patient_slices(patient_id, flair_dir, labels_csv_dir):
    flair_path = os.path.join(flair_dir, f"{patient_id}_flair.nii")
    labels_csv_path = os.path.join(labels_csv_dir, f"{patient_id}_flair.nii_labels.csv")

    flair_img = load_normalized_img(flair_path)
    labels_df = pd.read_csv(labels_csv_path)

    X_slices = [flair_img[:, :, i] for i in range(flair_img.shape[2])]
    Y_slices = labels_df["label"].values

    slice_indices = np.arange(len(X_slices)) / (len(X_slices) - 1)

    X_slices = np.expand_dims(X_slices, axis=-1)
    slice_indices = np.expand_dims(slice_indices, axis=-1)

    return [np.array(X_slices), np.array(slice_indices)], np.array(Y_slices)


def train(model, train_dir, train_label_dir, model_path, num_epochs, batch_size):
    train_patients = sorted([f.replace("_flair.nii", "") for f in os.listdir(train_dir) if f.endswith(".nii")])
    print(f"Train Patients: {len(train_patients)}")
    for epoch in range(num_epochs):
        random.shuffle(train_patients)  # Shuffle patient order at each epoch
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for idx, patient_id in enumerate(train_patients):
            print(f"Training on patient {idx + 1}/{len(train_patients)}: {patient_id}")

            X_train, Y_train = load_patient_slices(patient_id, train_dir, train_label_dir)
            model.fit(X_train, Y_train, epochs=1, batch_size=batch_size, verbose=1)

    model.save(model_path)
    print(f"\nModel saved at: {model_path}")



def evaluate_model(model, test_dir, test_label_dir):
    test_patients = sorted([f.replace("_flair.nii", "") for f in os.listdir(test_dir) if f.endswith(".nii")])
    print(f"\nEvaluating on {len(test_patients)} test patients...")

    all_X_test, all_Y_test = [], []

    for idx, patient_id in enumerate(test_patients):
        print(f"Testing on patient {idx + 1}/{len(test_patients)}: {patient_id}")

        X_test, Y_test = load_patient_slices(patient_id, test_dir, test_label_dir)

        all_X_test.append(X_test)
        all_Y_test.append(Y_test)

    # Concatenate test data across all patients
    all_X_test = [np.concatenate([x[i] for x in all_X_test], axis=0) for i in range(len(all_X_test[0]))]
    all_Y_test = np.concatenate(all_Y_test, axis=0)

    # Get model predictions
    Y_pred_probs = model.predict(all_X_test)
    Y_pred = (Y_pred_probs > 0.5).astype(int)

    # Compute evaluation metrics
    test_acc = accuracy_score(all_Y_test, Y_pred)
    test_precision = precision_score(all_Y_test, Y_pred, zero_division=0)
    test_recall = recall_score(all_Y_test, Y_pred, zero_division=0)
    test_f1 = f1_score(all_Y_test, Y_pred, zero_division=0)

    # Print evaluation results
    print("\nFinal Test Metrics:")
    print(f"  Accuracy  : {test_acc:.4f}")
    print(f"  Precision : {test_precision:.4f}")
    print(f"  Recall    : {test_recall:.4f}")
    print(f"  F1-Score  : {test_f1:.4f}")


