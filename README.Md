## A Comparitive Study On Multi-Stage Fine tuning Approaches for Diabetic Retinopathy detection

**Diabetic Retinopathy (DR)** is the leading cause of preventable blindless and is said to affect up to 700 million people by 2045. Diabetic retinopathy is an example of diabeitc mellitus (diabetes) and is a disease that essentially affects the blood vessels of the retina. If left undetected, it can lead to vision impairment or even blindness. Early diagnosis is crucial, and deep learning models have shown promise in assisting automated detection from retinal images. This project entails a Deep learning approach to enable early and accurate detection. 

The motivations for this project :
- I've been Highly interested in the field of Deep learning and its approaches within the medical field.

- Patients today often face delays in diagnosis due to limited access to specialists and time-consuming manual screening processes. Automated detection systems like the one proposed in this project can help accelerate diagnosis, reduce workload on medical professionals, and potentially improve early intervention outcomes.

![Comparison between healthy and a DR eye](https://www.lei.org.au/wp-content/uploads/2022/10/shutterstock_328135913-scaled.jpg)
## Table of contents 
4. [Dataset](#dataset)
The dataset used in this project is known as the Eyepacs dataset provided by WILL CUKIERSK in the [2015 kaggle competition](https://www.kaggle.com/c/diabetic-retinopathy-detection). It contains high-resolution retinal images labeled by DR severity levels ranging from 0 (No DR) to 4 (Proliferative DR). 

All images were taken from patients and includes both the left and right eye of the patient. High resolution images and images containing noise were captured,the steps taken in the [preprocessing section](#preprocessing) show how I was able to solve this problem.


    - [Structure](#structure)
    The dataset consists of a zip file for all training examples and testing examples. Howvever due to the test set having private lables. We opted to only use the training images as our full dataset. The train folder consisted of 35,126 images that contained labelling, hence this was used as the dataset in the study.

    - [Preprocessing](#preprocessing)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [Class Distribution](#class-distribution)
    - [Sample Images](#sample-images)
    - [Observations](#observations)
6. [Modeling Approach](#modeling-approach)
    - [Baseline Model](#baseline-model)
    - [Fine-Tuning Strategies](#fine-tuning-strategies)
    - [Model Architectures](#model-architectures)
7. [Training Details](#training-details)
    - [Loss Function](#loss-function)
    - [Optimizer and Scheduler](#optimizer-and-scheduler)
    - [Early Stopping and Checkpoints](#early-stopping-and-checkpoints)
8. [Evaluation](#evaluation)
    - [Metrics Used](#metrics-used)
    - [Validation and Test Results](#validation-and-test-results)
    - [Confusion Matrix](#confusion-matrix)
9. [Visualizations](#visualizations)
10. [Challenges and Solutions](#challenges-and-solutions)
11. [Conclusion](#conclusion)
12. [Future Work](#future-work)
13. [How to Run the Project](#how-to-run-the-project)
14. [References](#references)
.
