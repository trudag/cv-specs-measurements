CV-Specs-Measurement
## Overview
The CV-Specs-Measurement project aims to enhance image and video processing capabilities using advanced machine learning models, specifically YOLOv8 for object detection. This project is structured into several modules, including data augmentation, model training, and deployment for both picture and video analysis.

Table of Contents
Project Structure
Installation
Usage
Data Augmentation
Model Training
Picture Analysis
Video Analysis
Docker
Contributing
License

# Installation

## Prerequisites
Python 3.9
Docker

## Clone the repository
git clone https://github.com/yourusername/cv-specs-measurement.git
cd cv-specs-measurement

## Install Dependencies



## Project structure
```
cv-specs-measurements
├─ Data_augmentation
│  ├─ augmentation_main.py
│  ├─ Example_markers
│  ├─ Sample_data
│  │  ├─ sample_images1024_holdout
│  │  └─ sample_images1024_training
│  └─ Supporting_code_aug
│     ├─ augutilities.py
│     ├─ unittests.py
│     └─ Unit_test_data
│        ├─ test_backgrounds
│        ├─ test_markers
│        └─ test_output
├─ Model_deployment
│  ├─ Models
│  │  └─ Two_classes_e100_r512_b16
│  │     ├─ CNN_visualisation.svg
│  │     ├─ Performance_metrics
│  │     │  ├─ F1_curve.png
│  │     │  ├─ PR_curve.png
│  │     │  ├─ P_curve.png
│  │     │  ├─ results.png
│  │     │  ├─ R_curve.png
│  │     │  └─ training_results.csv
│  │     ├─ Training_localisation_examples
│  │     │  ├─ train_batch7382.jpg
│  │     │  ├─ val_batch0_labels.jpg
│  │     │  ├─ val_batch0_pred.jpg
│  │     │  └─ val_batch1_labels.jpg
│  │     └─ Weights
│  │        ├─ best.pt
│  │        └─ second_best.pt
│  ├─ Supporting_code_depl
│  │  ├─ deplutilities.py
│  │  ├─ picture_unittests.py
│  │  ├─ Unit_test_data
│  │  │  └─ 53986.png
│  │  └─ video_unittests.py
│  ├─ Task1_Picture_analysis
│  │  ├─ picture_main.py
│  │  └─ Test_data
│  │     ├─ input_frame.jpg
│  │     └─ Output
│  │        └─ output_input_frame.jpg
│  └─ Task2_Frame_identification
│     ├─ Test_data
│     │  ├─ Output
│     │  │  ├─ frame138.jpg
│     │  │  └─ output_results.json
│     │  └─ video.mp4
│     └─ video_main.py
├─ Model_training
│  ├─ model_spec.yaml
│  ├─ Sample_development_data
│  │  └─ datasets
│  │     ├─ holdout_data
│  │     └─ training_data
│  ├─ Supporting_code_train
│  │  └─ trainutilities.py
│  ├─ training_main.py
│  └─ yolov8s.yaml
├─ README.md
└─ requirements.txt

```
