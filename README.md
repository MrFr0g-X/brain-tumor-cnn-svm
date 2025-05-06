# Brain Tumor Classification Using CNN + SVM Hybrid Models

This repository contains Jupyter notebooks implementing various approaches to brain tumor classification from MRI images using combinations of Convolutional Neural Networks (CNNs) and Support Vector Machines (SVMs).

## Project Overview

These notebooks explore hybrid approaches combining deep learning feature extraction with classical machine learning classification for brain tumor detection and classification. The goal is to leverage both the feature extraction capabilities of CNNs and the classification strength of SVMs. Most approaches follow a similar pattern:

1. Use a CNN (often pre-trained) to extract high-level features from MRI images
2. Select/reduce features using techniques like PCA, Random Forest importance, etc.
3. Train SVM classifier on extracted features
4. Apply XAI (Explainable AI) techniques like LIME, SHAP, Grad-CAM to interpret results

This hybrid approach typically outperforms both standalone CNN and standalone SVM methods for medical image classification tasks.

## Repository Structure

The repository is organized by student contributions:

### Ahmed Ayman (202200168)

- `brain-tumor-classification-optimized-svm.ipynb`: Implements an optimized SVM approach
- `brain-tumor-classification-using-hybrid-csvm.ipynb`: Hybrid CNN-SVM approach
- `phase-2-paper.ipynb`: Initial implementation

### Hothifa Hamdan (202201792)

- `Hothifa_Hamdan_Paper1_Yeh_ConvSVM.ipynb`: Implementation based on Yeh paper
- `Hothifa_Hamdan_Paper8_Shanjida_ParallelCNN_v5_DebugSVM.ipynb`: Parallel CNN approach with debug steps for SVM
- `ResNet50 and Grad-CAM_FULL NOTEBOOK_Hothifa Hamdan_202201792.ipynb`: ResNet50 with Grad-CAM visualization

### Mazen Khaled (202201534)

- `a-comparison-between-support-vector-machine-svm.ipynb`: Comparative analysis of SVM approaches
- `mazen-khaled-paper-duggani-cnnensemblesvm_final.ipynb`: Implementation based on Duggani paper
- `mazen-khaled-paper12-yan-explainableframework_final.ipynb`: Explainable framework based on Yan paper

### Toka Mokhtar (202201920)

- `Conv_SVM_MNIST (3).ipynb`: CNN-SVM approach tested on MNIST dataset
- `notebookd2f7dc33aa (1).ipynb`: Implementation for brain tumor detection
- `toka-1.ipynb`: Feature selection approaches for CNN-SVM

## Dataset

The notebooks use the Brain Tumor MRI Dataset from Kaggle, which contains MRI images classified into four categories:

- Glioma
- Meningioma
- No tumor
- Pituitary tumor

## Explainable AI (XAI) Techniques

Several XAI techniques are implemented to interpret model decisions:

- **Grad-CAM**: Visualizes areas of the image that influenced the CNN's decision
- **LIME**: Provides local interpretable explanations for individual predictions
- **SHAP**: Calculates feature importance using Shapley values
- **Partial Dependence Plots**: Shows how features influence predictions
- **Feature Importance Analysis**: Identifies which features contribute most to classification

## Requirements

The notebooks require the following libraries:

- TensorFlow/Keras
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- LIME
- SHAP
- OpenCV

## Acknowledgments

This work was completed as part of DSAI 305 course project at Zewail City University.