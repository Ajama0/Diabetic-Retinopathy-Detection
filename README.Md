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
   - [Baseline Model](#baseline-model)
   - [Fine-Tuning Strategies](#fine-tuning-strategies)
   - [Model Architectures](#model-architectures)
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

The images provided by the authors were taken under poor lighting conditions which made the blood vessels and other symptoms of DR difficult to notice. High boost filtering was used to enhance the contrast of retinal images, making subtle features such as microaneurysms, hemorrhages, and blood vessels more prominent for improved detection. (See Paper for in depth reasoning). Black_images.py was ran however there were no fully black images in the original dataset. 


1. Run [`preprocess_2.py`](./scripts/preprocess_2.py) on the original training images – output will be a directory of preprocessed images.  
2. Run [`convert_to_binary_classification.py`](./scripts/convert_to_binary_classification.py) on the original CSV to transform all labels to either 0 or 1 (No DR vs DR).




An single output image after preprocessing should look like the below

!["Preprocessed-Image"](DR_preprocessed.png)






