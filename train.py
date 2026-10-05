"""Train a MobileNetV2 transfer-learning CNN for Chest X-ray Pneumonia detection.

Usage:
    python train.py --data_dir data/chest_xray
(data_dir must contain train/ and test/, each with NORMAL/ and PNEUMONIA/)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42
CLASSES = ["NORMAL", "PNEUMONIA"]   # NORMAL = 0, PNEUMONIA = 1


def load_ds(path, subset=None, shuffle=True):
    ds = keras.utils.image_dataset_from_directory(
        path, labels="inferred", label_mode="binary", class_names=CLASSES,
        image_size=IMG_SIZE, batch_size=BATCH, shuffle=shuffle, seed=SEED,
        validation_split=0.2 if subset else None, subset=subset)
    return ds.prefetch(tf.data.AUTOTUNE)


def build_model():
    base = keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,),
                                          include_top=False, weights="imagenet")
    base.trainable = False
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = layers.RandomRotation(0.05)(inputs)          # augmentation (training only)
    x = layers.RandomZoom(0.1)(x)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)  # MobileNetV2 expects [-1, 1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # P(PNEUMONIA)
    return keras.Model(inputs, outputs), base


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/chest_xray")
    ap.add_argument("--epochs_head", type=int, default=5)
    ap.add_argument("--epochs_finetune", type=int, default=5)
    args = ap.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")

    train_ds = load_ds(train_dir, "training")
    val_ds = load_ds(train_dir, "validation")
    test_ds = load_ds(test_dir, shuffle=False)

    # Dataset is imbalanced (more pneumonia than normal) -> class weights
    n0 = len(os.listdir(os.path.join(train_dir, "NORMAL")))
    n1 = len(os.listdir(os.path.join(train_dir, "PNEUMONIA")))
    total = n0 + n1
    class_weight = {0: total / (2 * n0), 1: total / (2 * n1)}
    print("Class weights:", class_weight)

    model, base = build_model()
    early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    print("\n--- Phase 1: train the new classifier head ---")
    compile_model(model, 1e-3)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head,
                   class_weight=class_weight, callbacks=[early])

    print("\n--- Phase 2: fine-tune the last 30 layers of MobileNetV2 ---")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    compile_model(model, 1e-5)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_finetune,
                   class_weight=class_weight, callbacks=[early])

    model.save("pneumonia_model.keras")

    # ---- Evaluate on the untouched test set ----
    print("\n--- Test evaluation ---")
    y_true = np.concatenate([y.numpy() for _, y in test_ds]).ravel().astype(int)
    probs = model.predict(test_ds).ravel()
    y_pred = (probs > 0.5).astype(int)
    report = classification_report(y_true, y_pred, target_names=CLASSES)
    acc = (y_true == y_pred).mean() * 100
    print(f"Test accuracy: {acc:.2f}%\n{report}")
    with open("results.txt", "w") as f:
        f.write(f"Test accuracy: {acc:.2f}%\n\n{report}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
    plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.close()

    acc_hist = h1.history["accuracy"] + h2.history["accuracy"]
    val_hist = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss_hist = h1.history["loss"] + h2.history["loss"]
    vloss_hist = h1.history["val_loss"] + h2.history["val_loss"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(acc_hist, label="train"); ax[0].plot(val_hist, label="val"); ax[0].set_title("Accuracy"); ax[0].legend()
    ax[1].plot(loss_hist, label="train"); ax[1].plot(vloss_hist, label="val"); ax[1].set_title("Loss"); ax[1].legend()
    plt.tight_layout(); plt.savefig("training_curves.png", dpi=150)
    print("Saved: pneumonia_model.keras, results.txt, confusion_matrix.png, training_curves.png")


if __name__ == "__main__":
    main()
