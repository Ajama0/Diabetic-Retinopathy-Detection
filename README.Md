## A Comparative Study on Multi-Stage Fine-Tuning Approaches for Diabetic Retinopathy Detection

**Diabetic Retinopathy (DR)** is the leading cause of preventable blindness and is projected to affect up to 700 million people by 2045. Diabetic retinopathy is a complication of diabetes mellitus that affects the blood vessels of the retina. If left undetected, it can lead to vision impairment or even blindness. Early diagnosis is crucial, and deep learning models have shown promise in assisting automated detection from retinal images. This project explores a deep learning approach to enable early and accurate detection.

### Motivations for this project
- I've been highly interested in the field of deep learning and its applications within the medical field.
- Patients today often face delays in diagnosis due to limited access to specialists and time-consuming manual screening processes. Automated detection systems like the one proposed in this project can help accelerate diagnosis, reduce workload on medical professionals, and potentially improve early intervention outcomes.

![Comparison between healthy and a DR eye](https://www.lei.org.au/wp-content/uploads/2022/10/shutterstock_328135913-scaled.jpg)

---

## Table of Contents

1. [Dataset](#dataset)
   - [Structure](#structure)
2. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Class Distribution](#class-distribution)
   - [Preprocessing](#preprocessing)
   - [Data Augmentation](#data-augmentation)
3. [Modeling Approach](#modeling-approach)
   - [Model architectures](#model-architectures)
   - [Fine-Tuning Strategies](#fine-tuning-strategies)
4. [Training Details](#training-details)
   - [Loss Function](#loss-function)
   - [Optimizer and Scheduler](#optimizer-and-scheduler)
   - [Early Stopping and Checkpoints](#early-stopping-and-checkpoints)
5. [Evaluation](#evaluation)
   - [Metrics Used](#metrics-used)
   - [Validation and Test Results](#validation-and-test-results)
   - [Confusion Matrix](#confusion-matrix)
6. [Visualizations](#visualizations)
7. [Challenges and Solutions](#challenges-and-solutions)
8. [Conclusion](#conclusion)
9. [Future Work](#future-work)
10. [How to Run the Project](#how-to-run-the-project)
11. [References](#references)

---

## Dataset

The dataset used in this project is known as the **EyePACS dataset**, provided by Will Cukierski in the [2015 Kaggle competition](https://www.kaggle.com/c/diabetic-retinopathy-detection). It contains high-resolution retinal images labeled by DR severity levels ranging from 0 (No DR) to 4 (Proliferative DR).

All images were taken from patients and include both the left and right eye. Some images were noisy or low contrast. The steps taken in the [preprocessing section](#preprocessing) explain how these issues were addressed.

### Structure

The dataset consists of a zip file containing all training and testing examples. However, due to the test set having private labels, we opted to only use the training images as our full dataset. The training folder contains 35,126 images with labels, which were used for training and evaluation in this study.

## Exploratory Data analysis

35,126 images are present in the dataset. There are 5 severity gradings from No DR - Proliferative DR, No-dr has the highest prevalence with 73% of the dataset being healthy(No DR). To mitigate this we turned the study into a 2 class problem to handle the large imbalance. Data augmentation techniques seen in the following sections helped to improve generalizability. 

Image below shows 2 sets of original training images across all severity gradings

!["original-images](original_images.png)



### Class Distribution 
![Class distribution](class-distribution.png)

### Preprocessing


As images contained large Black borders which contains irrelevant information that can alter the way a deep neural network extracts relevant information, we implemented a cropping algorithm with a threshold value which aims to only capture the retina from each image.
Consistent sizing of the images were something the authors did not handle therefore images were scaled to ensure the network sees a consistent input. 

The images provided by the authors were taken under poor lighting conditions which made the blood vessels and other symptoms of DR difficult to notice. High boost filtering was used to enhance the contrast of retinal images, making subtle features such as microaneurysms, hemorrhages, and blood vessels more prominent for improved detection. See ['DR.pdf](./Paper/DR.pdf) for in depth reasoning.[`black_images.py`](./scripts/black_images.py)was ran however there were no fully black images in the original dataset. 


1. Run [`preprocess_2.py`](./scripts/preprocess_2.py) on the original training images – output will be a directory of preprocessed images.  
2. Run [`convert_to_binary_classification.py`](./scripts/convert_to_binary_classification.py) on the original CSV to transform all labels to either 0 or 1 (No DR vs DR).




An single output image after preprocessing should look like the below

!["Preprocessed-Image"](DR_preprocessed.png)


### Data Augmentation
Online Data augmentation was used to allow the model to see variations of each image per EPOCH, this allows for better generalizability and essentially mitigates the problem of having to oversample and store more images on the hard disk. 

The data augmentation used was 
- Random Horizontal flio
- Random Rotation
- Random Vertical Flip
- Standardization was done on the train set giving a mean of 0 and unit variance of 1


After all the preprocessing steps, you should end up with a new dataset. 
<p align="center">
  <img src="New-Dataset.png" alt="New Dataset Preview" width="400"/>
</p>




## Modelling Approach
Three pretrained architectures were used preventing us from training the model from scratch with random weights and biases. 
Inception v3, ResNet50 and DenseNet121 were used, all 3 architectures were modified in which Dropout layers, Activation functions and a Classifier layer were added, This was added to the improve generalizability and also map the feature maps to our 2 classes. The !["DR.pdf"](./Paper/DR.pdf) goes in depth explaining the reasoning for these modifications and the parameters used. 

### Model architectures

<table>
  <tr>
    <td align="center">
      <img src="resnet50.png" width="200"/><br/>
      <strong>ResNet50</strong>
    </td>
    <td align="center">
      <img src="inceptionv3.png" width="200"/><br/>
      <strong>InceptionV3</strong>
    </td>
    <td align="center">
      <img src="densenet121.png" width="200"/><br/>
      <strong>DenseNet121</strong>
    </td>
  </tr>
</table>



### Fine-Tuning Strategies
Instead of training the model from scratch, we leveraged Transfer learning which allowed us to transfer the knowledge gained from one domain(image net) to ours. This meant the pretrained parameters were used however optimized for our Task at hand. All 3 networks underwent consistant modifications instead of Inception v3 which already contained a Dropout layer. Networks were trained using a multi-stage process in which the classifier head was trained for N epochs to prevent initial weight distortion then all layers were set to trainable. With trial and error of different methods, this yielded the best results.

Slight overview

- ✅ Transfer Learning: Used pretrained weights from ImageNet to avoid training from scratch, which reduced training time and improved convergence.
- 🎯 Domain Adaptation: Fine-tuned the pretrained models specifically for retinal image classification, adjusting parameters to our binary DR task.
- 🧱 Multi-Stage Training:
First stage: Trained only the classifier head to avoid disrupting pretrained feature representations.
Second stage: Unfroze all layers and fine-tuned the entire network to adapt low-level features to our dataset.
- 🔄 Architecture Adjustments: Added dropout layers and fully connected heads (except for InceptionV3 which already includes dropout) to improve generalization and reduce overfitting.
- 🔍 Empirical Tuning: Found this strategy to outperform other methods through trial-and-error experiments.










