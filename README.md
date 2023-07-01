# Breast Cancer Detection

The goal of this project is to identify breast cancer. You'll train your model with screening mammograms obtained from regular screening.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Dataset](#dataset)
5. [Model Architecture](#model-architecture)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Prediction](#prediction)
9. [License](#license)

## Introduction

The goal of this project is to identify breast cancer. You'll train your model with screening mammograms obtained from regular screening.

## Installation

Before running the code, please make sure you have the following dependencies installed:

- python-gdcm
- pydicom
- pylibjpeg
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- tensorflow
- medpy
- joblib
- tqdm

You can install these dependencies using the following command:

pip install -r requirements.txt

## Usage

To use this code, follow these steps:

1. Ensure that the dataset files are available in the specified directories (`train_images`, `train.csv`, `test_images`, `test.csv`).
2. Update the paths to the dataset files in the code as necessary.
3. Run the code using the command `python main.py`.
4. The code will preprocess the images, train the CNN model, and generate predictions for the test data.
5. The predictions will be stored in a CSV file named `submission.csv`.

## Dataset

The dataset used for breast cancer detection consists of mammogram images. The training dataset is provided in the `train_images` directory, and the corresponding labels are stored in the `train.csv` file. The test dataset is provided in the `test_images` directory, and the associated metadata is available in the `test.csv` file.

## Model Architecture

The CNN model used for breast cancer detection consists of several convolutional layers with ReLU activation, followed by max-pooling layers. The architecture is as follows:

1. Conv2D (32 filters, 3x3 kernel, ReLU activation)
2. MaxPooling2D (2x2 pool size)
3. Conv2D (64 filters, 3x3 kernel, ReLU activation)
4. MaxPooling2D (2x2 pool size)
5. Conv2D (64 filters, 3x3 kernel, ReLU activation)
6. Flatten
7. Dense (64 units, ReLU activation)
8. Dense (1 unit, sigmoid activation)

The model is compiled with the Adam optimizer, binary cross-entropy loss, and accuracy metric.

## Training

The training process involves the following steps:

1. Loading and preprocessing the training images from the `train_images` directory.
2. Splitting the preprocessed images and labels into training and validation sets.
3. Creating the CNN model and compiling it with the necessary configurations.
4. Training the model on the training set with a specified number of epochs and batch size.
5. Saving the best model weights based on validation accuracy using checkpoints.
6. Loading the saved model weights for evaluation and prediction.

## Evaluation

The trained model is evaluated on the validation set to measure its performance. The evaluation includes calculating the test loss and accuracy. The results are printed on the console.

## Prediction

After training and evaluation, the model is used to generate predictions for the test dataset. The test images are preprocessed and passed through the trained model. The predicted probabilities for breast cancer presence are obtained.


## License

Information about the license under which your project is published.

