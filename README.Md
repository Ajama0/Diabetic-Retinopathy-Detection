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
   - [Preprocessing](#preprocessing)
2. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Class Distribution](#class-distribution)
   - [Sample Images](#sample-images)
   - [Observations](#observations)
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
