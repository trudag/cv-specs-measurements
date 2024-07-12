![image](https://github.com/user-attachments/assets/bb57db2e-9754-45d8-837e-913a288f9f9f)CV-Specs-Measurement
## Overview
The `CV-Specs-Measurement` is an example of the application of convolutional neural networks to the localisation of custom fiducial markers. The project heavily relies on YOLOv8. This project is structured into several modules, including data augmentation, model training, and deployment for both picture and video analysis.

## Table of Contents
1. [General structure](#general-structure)
2. [Installation](#installation)
3. [Usage](#usage)
   - [Data Augmentation](#data-augmentation)
   - [Model Training](#model-training)
   - [Model Deployment](#model-deployment)
      - [Picture Analysis](#picture-analysis)
      - [Video Analysis](#video-analysis)
      - [Performance Metrics](#performance-metrics)

# General structure
```
cv-specs-measurements
├─ Data_augmentation (data augmentation algorithm that creates training data)
├─ Model_training (training CNN on the augmented data)
├─ Model_deployment (trained model for fiducial marker identification)
│  ├─ Models (weights of CNN)
│  ├─ Task1_Picture_analysis (first part of the technical assessment)
│  └─ Task2_Frame_identification (second part of the technical assessment)
└─ Test (video for testing and validation)
```
The Model_deployment part of the project is containerised within two Docker images: one with the algorithm for the identification of markers on the photo, and one for the identification of the neutral face pose in a video using markers. The main scripts also run several automatic unit tests using native Python functionality to ensure the integrity of the used functions. The code throughout also includes basic data validation functionality.

# Installation

## Prerequisites
Python 3.9
Docker

## Clone the repository
```bash
git clone https://github.com/trudag/cv-specs-measurements.git
cd cv-specs-measurements
```

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Build and Run Docker Containers
Instal the appropriate version of Docker. Within CMD navigate to the project folder and execute (on Windows):
1. Build the Docker images:
   ```bash
   docker-compose build
   ```
2. Run the Docker containers:
   ```bash
   docker-compose up
   ```

# Usage

## Data Augmentation
The Data Augmentation module is designed to generate synthetic training and holdout datasets by applying various transformations to the input images and markers. This helps create a model invariant for different types of noise.

### Sample data
You can find sample pictures of faces used for training in the Sample_data folder. During the training almost 2k unique pictures were used.
```bash
cd Data_augmentation/Sample_data
```
The examples of augmented data can be found within the Models folder.
```bash
cd Model_deployment/Models/Two_classes_e100_r512_b16/Training_localisation_examples
```

### Example Usage

1. **Set the Configuration Parameters**

   Edit the constants in the `augmentation_main.py` script to specify the desired paths and configuration options.

2. **Run the Data Augmentation Pipeline**

   Execute the script to generate the augmented datasets:
   ```bash
   python augmentation_main.py
   ```


## Model Training
The Model Training module is designed to train a YOLO model using the provided configuration and dataset.


### Example Usage

1. **Set the Configuration Parameters**

   Edit the constants in the `training_main.py` script to specify the desired paths and configuration options.

2. **Run the Training Script**

   Execute the script to start training the model:
   ```bash
   python training_main.py
   ```


## Model Deployment
### Picture Analysis
To identify markers on images (only .png and .jpg are allowed):
1. Navigate to the `Model_deployment/Task1_Picture_analysis` directory.
2. Run the `picture_main.py` script with the appropriate arguments.

#### Arguments for `picture_main.py`

- `--model`: Path to the model weights.
- `--input_dir`: Directory containing input images (only .jpg and .png are supported).
- `--output_dir`: Directory to save output images.

#### Running with arguments (Docker)
For picture analysis:
```bash
docker run --rm -v /path/to/models:/app/Model_deployment/Models -v /path/to/test:/app/Test cv-specs-measurement-picture-analysis --model /app/Model_deployment/Models/Two_classes_e100_r512_b16/Weights/best.pt --input_dir /app/Test/ --output_dir /app/Test/Output/
```

### Video Analysis
The algorithms uses the CNN to identify round markers on the measurement frame and to determine the frame with the 'most rectangle' marker positions. By 'most rectagle' is mean that the angles in corner are as close to 90 degrees as possible.

To analyse videos:
1. Navigate to the `Model_deployment/Task2_Frame_identification` directory.
2. Run the `video_main.py` script with the appropriate arguments.

#### Arguments for `video_main.py`

- `--model`: Path to the model weights.
- `--input_video`: Path to the input video file.
- `--output_video`: Path to save the output video with detections (may be problems with saving as .avi, but will play it if show-video is present).
- `--output_json`: Path to save the output frame data (marker coordinates) to JSON results.
- `--process_video`: Flag to extract frame data to the video and save it into a JSON (takes a loooong time; don't run if you've already processed the video)
- `--show_video`: Flag to display the video with detections.

#### Running with Arguments (Docker)
For video analysis:
```bash
docker run --rm -v /path/to/models:/app/Model_deployment/Models -v /path/to/test:/app/Test cv-specs-measurement-video-analysis --model /app/Model_deployment/Models/Two_classes_e100_r512_b16/Weights/best.pt --input_video /app/Test/video.mp4 --output_video /app/Test/Output/output_video_with_detections.avi --output_json /app/Test/Output/output_results.json --process_video --show_video
```
#### Performance metrics
The trained model has pretty decent descrimination performance. You can examine the metrics within the Models folder.   
```bash
cd Model_deployment/Models/Two_classes_e100_r512_b16/Performance_metrics
```

