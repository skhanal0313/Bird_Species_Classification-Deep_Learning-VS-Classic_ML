
# Fine-Grained Bird Species Classification using ML and CNN

![MATLAB](https://img.shields.io/badge/MATLAB-R2023b-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-CaltechUCSD--Birds--200--2011-orange)

## 🐦 Project Overview

This project explores and compares classic machine learning (SIFT/HOG + SVM/Random Forest) and deep learning (CNN) approaches for fine-grained bird species recognition using the Caltech-UCSD Birds-200-2011 dataset. With over 11,000 bird images across 200 species, the models were trained on both full images and bounding box-cropped inputs to enhance feature learning.

## 📂 Dataset

- **Source**: [Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
- **Size**: 11,788 images
- **Classes**: 200 bird species
- **Features**: Bounding box, class labels, segmentation masks

## 🧠 Models and Experiments

- **Classic ML Models**:  
  - SIFT + SVM  
  - HOG + SVM  
  - HOG + Random Forest

- **Deep Learning**:  
  - Custom-built CNN  
  - ResNet-50 (transfer learning for comparison)

- **Inputs Used**:  
  - Whole Image  
  - Bounding Box Cropped Image  

- **Cross-validation**:  
  - 5-fold validation used for bounding box CNN experiments


## 🚀 Getting Started

### Prerequisites
- MATLAB R2023b or later with:
  - Deep Learning Toolbox
  - Computer Vision Toolbox
  - Statistics and Machine Learning Toolbox

### How to Run
1. Open `.mlx` files (e.g., `experiment_4_BOUNDING_BOX_CNN_200Class_Best.mlx`) in MATLAB.
2. Run each experiment script to replicate model training and testing.
3. Ensure dataset path and image folders are correctly configured before running.

## 📁 File Descriptions

- `experiment_1_HOG_SVM_200Class.mlx`: Classic ML on whole images
- `experiment_2_CNN_200Class_Best.mlx`: CNN on whole images
- `experiment_3_HOG_SVM_200Class.mlx`: HOG + SVM on bounding boxes
- `experiment_4_BOUNDING_BOX_CNN_200Class_Best.mlx`: CNN on cropped inputs
- `experiment_4_BOUNDING_BOX_ResNet50_200Class_Best.mlx`: Transfer learning baseline
- `experiment_5_BOUNDING_BOX_CNN_5folds_200Class_Best.mlx`: CNN with 5-fold CV

## 👨‍🔬 Contributors

- **Sujan Khanal** – u3258630@uni.canberra.edu.au

## 📄 License

MIT License – see `LICENSE` for details.

## 📚 References

- Caltech-UCSD Birds-200-2011 Dataset
- Lowe, D.G., "Distinctive image features from scale-invariant keypoints"
- Dalal, N., & Triggs, B., "Histograms of oriented gradients for human detection"
- He, K., et al. "Deep Residual Learning for Image Recognition"
