# Gender Detection using CNN

## Project Overview

This project builds a deep learning model that predicts the gender (man / woman) of the person in a cropped face image, using a Convolutional Neural Network (CNN) trained from scratch.

The project was completed as part of the **DataMites AI Engineer Program (Internship Project)**.

The primary objective was to explore and analyze a set of **2,307 face images** across 2 classes, preprocess them for a CNN, train and evaluate the model, and provide a simple function that predicts gender and confidence for any new face image.

## Project Objectives

- Explore the image data: class balance, image sizes, aspect ratios, and brightness.
- Detect data quality issues such as non-image files, low-resolution images, and duplicates.
- Preprocess images with resizing, normalization, and data augmentation.
- Build and train a custom CNN to classify faces as man or woman.
- Evaluate the model using accuracy, precision, recall, F1 score, and a confusion matrix.
- Save the trained model and predict gender on new images.

## Technologies & Libraries

- Python
- Jupyter Notebook
- PyTorch
- Torchvision
- Pillow (PIL)
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Project Workflow

### 1. Data Overview
- Loaded only `.jpg` files from the `man` and `woman` folders (hidden `.DS_Store` files were ignored).
- Found **1,173 man** (50.8%) and **1,134 woman** (49.2%) images, so the classes are balanced and no resampling was needed.
- Image widths ranged from **24 to 2,269 px**, with a median of 160 px.
- Displayed random sample images from each class.

### 2. Data Analysis
- Measured width, height, aspect ratio, and brightness for every image, and used MD5 hashes to find duplicate files.
- Used histograms of width and aspect ratio and a brightness boxplot by gender.

Key findings:
- Median image size is about 160x214 for both genders, so image size does not give away the label.
- Most images are portrait (median aspect ratio ~0.75).
- 47 images are under 50 px wide. These low-resolution faces are harder to classify.
- Women's images are slightly brighter (median 133.9 vs 125.4).
- 4 duplicate files were found, and 1 identical image is stored in both the man and woman folders (label noise). These are too few to affect results.

### 3. Image Preprocessing
- Resized all images to 96x96.
- Applied augmentation to the training data only: horizontal flip, small rotation (10 degrees), and colour jitter.
- Converted images to tensors and normalized pixel values to [-1, 1].
- Split the data 80/20 into **1,845 training** and **462 test** images.

### 4. CNN Model Development

A custom CNN was built and trained from scratch:

- 4 convolution blocks: Conv 3x3, BatchNorm, ReLU, MaxPool (32, 64, 128, 256 filters)
- Global average pooling, Dropout 0.5, and a dense layer with 2 outputs
- **389,890** trainable parameters
- Input: 96x96 RGB images

### 5. Model Evaluation

The model was evaluated on 462 unseen test images using accuracy, precision, recall, F1 score, and a confusion matrix. Training loss and train/test accuracy curves were also plotted.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Man | 0.978 | 0.966 | 0.972 | 235 |
| Woman | 0.965 | 0.978 | 0.972 | 227 |
| **Overall accuracy** | | | **0.972** | **462** |

| Metric | Value |
|---|---:|
| Train accuracy | 98.3% |
| Test accuracy | **97.2%** (449 / 462 correct) |
| Misclassified images | 13 (8 men predicted as women, 5 women predicted as men) |

**Best Performing Model: Custom 4-Block CNN**

The CNN reached **97.2% test accuracy** with balanced results for both classes (F1 0.972 each). The gap between train and test accuracy is only 1.1 points, so the model generalizes well. It is light (~390k parameters), so it can run quickly even on a CPU.

### 6. Prediction on New Images
- Saved the trained model as `gender_cnn.pth`.
- Wrote a `predict(path)` function that returns the predicted gender and its confidence for any face image.
- Displayed predictions on random test images, with correct predictions in green and wrong ones in red.

## Key Findings

- A simple CNN trained from scratch reached **97.2%** test accuracy on face images.
- Both genders were predicted equally well: man recall 0.966 and woman recall 0.978.
- Train (98.3%) and test (97.2%) accuracy are close, so augmentation and dropout kept overfitting low.
- Test accuracy jumped around in early epochs (for example 0.935 at epoch 8 and 0.807 at epoch 9) but settled once the learning rate dropped, finishing at 0.972.
- Image size and aspect ratio were similar for both classes, so they do not leak the label.
- The model expects a cropped face. For full photos, a face detector (e.g. OpenCV Haar cascade) should crop the face first.

## Challenges and Solutions

Some of the main challenges encountered included:

- **Varying image sizes:** Images ranged from 24 px to 2,269 px wide, so all were resized to 96x96 for a fixed CNN input.
- **Brightness difference between classes:** Used colour jitter so the model does not use lighting as a shortcut.
- **Unstable test accuracy in early epochs:** Used a OneCycle learning rate schedule for fast learning early and stable results at the end.
- **Cropped faces only:** Documented that full photos need a face detector to crop the face first.

## Disclaimer

This project was completed for educational and portfolio purposes as part of the DataMites AI Engineer Program. The results are based on a small set of cropped face images and should not be interpreted as a validated production gender detection system.

## Author

**Suchitra Thanumalayan**

Data Science | Machine Learning | Artificial Intelligence
