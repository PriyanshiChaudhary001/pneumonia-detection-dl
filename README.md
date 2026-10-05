# Pneumonia Detection from Chest X-rays (Deep Learning)

An educational deep-learning project that uses a Convolutional Neural
Network (CNN) with transfer learning to classify chest X-ray images into
two classes: **NORMAL** and **PNEUMONIA**.

The model uses **MobileNetV2**, pre-trained on ImageNet. It reports a
predicted class and confidence, along with precautionary information.
This project is for learning and experimentation only and is **not
suitable for medical diagnosis or treatment decisions**.

## Features

-   CNN-based image classification with MobileNetV2 transfer learning.
-   Binary classification: `NORMAL` or `PNEUMONIA`.
-   Two-stage training: train the classifier head, then fine-tune the
    last 30 MobileNetV2 layers.
-   Class weights to help account for imbalance between the two classes.
-   Training outputs and evaluation results, as supported by the
    training script.

## Dataset

This project uses the [Chest X-Ray Images (Pneumonia) dataset on
Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

The dataset is **not included** in this repository. The expected
directory layout is:

``` text
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

Make sure the extracted dataset follows the directory structure expected
by the training script. Do not assume a downloaded archive will extract
to exactly this layout without checking it.

## Requirements

-   Python 3
-   Packages listed in `requirements.txt`
-   Google Colab with GPU runtime is recommended by this project for
    training

Install dependencies from the project directory:

``` bash
pip install -r requirements.txt
```

For Google Colab, select **Runtime → Change runtime type → GPU** before
training, if a GPU runtime is available.

## Dataset Setup in Google Colab

The original workflow uses the Kaggle API. Place your Kaggle API
credentials file (`kaggle.json`) in the Colab session securely. Do not
commit credentials or upload them to GitHub.

Install the Kaggle CLI:

``` python
!pip install -q kaggle
```

Create the Kaggle configuration directory and copy your credentials
file:

``` python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

Download and extract the dataset:

``` python
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
!unzip -q chest-xray-pneumonia.zip -d data
```

After extraction, inspect the folders and confirm that the dataset
directory matches the expected layout above.

## Training

From the project directory, run:

``` bash
python train.py --data_dir data/chest_xray
```

The documented training process has two phases:

1.  Train the new classifier head.
2.  Fine-tune the last 30 layers of MobileNetV2.

Class weights are used to help handle the imbalance between `NORMAL` and
`PNEUMONIA` images. Refer to the training script and its console output
for the exact training configuration and generated files.

## Prediction

Run prediction on an image using:

``` bash
python predict.py --image sample_xray.jpeg
```

Replace `sample_xray.jpeg` with the path to your own test image. Use an
image format supported by the prediction script.

## Results

The project README's original results section uses a placeholder.
Replace it only with the measured test accuracy reported by your actual
run.

``` text
Test accuracy: Add measured value from results.txt
```

The project may also produce the following visual outputs, depending on
the training and prediction scripts:

-   `training_curves.png`
-   `confusion_matrix.png`
-   `prediction_output.png`

Only include generated result images in the repository when they exist
and are appropriate to share. A result on one dataset does not guarantee
performance on other images or clinical settings.

## Project Structure

The key files documented for this project include:

``` text
pneumonia-detection-dl/
├── requirements.txt
├── train.py
├── predict.py
├── data/                 # Dataset, not included in the repository
├── results.txt           # Generated training/evaluation results
├── training_curves.png   # Generated output, if produced
├── confusion_matrix.png  # Generated output, if produced
└── prediction_output.png # Generated output, if produced
```

The exact files generated depend on the implementation and successful
completion of the scripts.

## Disclaimer

**Educational project only. This is not a medical device and must not be
used to diagnose pneumonia, rule out illness, or guide treatment.**

Model predictions and confidence scores are not medical conclusions.
Chest X-ray interpretation requires qualified healthcare professionals
and clinical context. Do not use this project to make health decisions.
Any health concern should be discussed with a qualified doctor.
