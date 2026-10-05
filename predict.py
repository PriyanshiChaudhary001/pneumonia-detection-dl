"""Analyze one chest X-ray: prediction, confidence %, risk level, precaution, cure.

Usage:
    python predict.py --image path/to/xray.jpeg
"""
import argparse
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

INFO = {
    "PNEUMONIA": {
        "precaution": "Avoid smoking and polluted air, get pneumonia and flu vaccines, wash hands often, rest and stay hydrated, keep away from sick people.",
        "cure": "Needs a doctor's diagnosis. Bacterial pneumonia is treated with antibiotics; viral with rest, fluids and antivirals when prescribed; severe cases need hospital care and oxygen.",
    },
    "NORMAL": {
        "precaution": "Maintain a healthy lifestyle, avoid smoking, exercise regularly and get recommended vaccines.",
        "cure": "No treatment needed. See a doctor if cough, fever or breathing difficulty continues.",
    },
}


def risk_level(p):
    if p < 0.5:
        return "Low"
    if p < 0.7:
        return "Moderate"
    if p < 0.9:
        return "High"
    return "Very High"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="pneumonia_model.keras")
    args = ap.parse_args()

    model = keras.models.load_model(args.model)
    img = keras.utils.load_img(args.image, target_size=(224, 224))   # converts to RGB
    arr = keras.utils.img_to_array(img)[None, ...]

    p_pneu = float(model.predict(arr, verbose=0)[0][0])
    label = "PNEUMONIA" if p_pneu >= 0.5 else "NORMAL"
    confidence = (p_pneu if label == "PNEUMONIA" else 1 - p_pneu) * 100
    info = INFO[label]

    lines = [
        f"Prediction            : {label}",
        f"Confidence            : {confidence:.2f}%",
        f"Pneumonia probability : {p_pneu * 100:.2f}%",
        f"Risk level            : {risk_level(p_pneu)}",
        "", "Precaution:", textwrap.fill(info["precaution"], 55),
        "", "Cure / Treatment:", textwrap.fill(info["cure"], 55),
        "", "NOTE: Educational project, NOT a medical diagnosis.",
    ]
    text = "\n".join(lines)
    print("\n" + text + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(np.array(img).astype("uint8")); ax[0].axis("off")
    ax[1].axis("off"); ax[1].text(0, 1, text, va="top", family="monospace", fontsize=10)
    plt.tight_layout(); plt.savefig("prediction_output.png", dpi=150)
    print("Saved: prediction_output.png (use this as your README screenshot)")


if __name__ == "__main__":
    main()
