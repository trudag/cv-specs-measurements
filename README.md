CV-Specs-Measurement
## Overview
The `CV-Specs-Measurement` is an example of the application of convolutional neural networks to the localisation of custom fiducial markers. The project heavily relies on YOLOv8. This project is structured into several modules, including data augmentation, model training, and deployment for both picture and video analysis.

## Table of Contents
1. [General structure](#general-structure)
2. [Installation](#installation)
3. [Usage](#usage)
   - [Data Augmentation](#data-augmentation)
   - [Model Training](#model-training)
   - [Picture Analysis](#picture-analysis)
   - [Video Analysis](#video-analysis)
4. [Docker](#docker)
5. [Contributing](#contributing)
6. [License](#license)

## General structure
```
cv-specs-measurements
├─ Data_augmentation (contains algorithm and sample supporting data for the data augmentation algorithm that creates training data)
├─ Model_training (contains the CNN algorithms that train using the augmented data)
├─ Model_deployment (contains the algorithms that use the trained model for fiducial marker identification)
│  ├─ Task1_Picture_analysis (first part of the technical assessment)
│  └─ Task2_Frame_identification (second part of the technical assessment)
└─ Test (contains the video for testing and validation)
```
The Model_deployment part of the project is containerised within two Docker images: one with the algorithm for identification of markers on the photo, and one for identification of the neutral face pose on a video using markers.

## Installation

### Prerequisites
Python 3.9
Docker

### Clone the repository
git clone https://github.com/yourusername/cv-specs-measurement.git
cd cv-specs-measurement

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Build and Run Docker Containers
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
Navigate to the `Data_augmentation` directory and run the `augmentation_main.py` script to generate augmented data.

## Model Training
Navigate to the `Model_training` directory and run the `training_main.py` script to start training the model using the provided `yolov8s.yaml` configuration file.

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
To analyze videos:
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




## Table of Contents
1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
   - [Data Augmentation](#data-augmentation)
   - [Model Training](#model-training)
   - [Picture Analysis](#picture-analysis)
   - [Video Analysis](#video-analysis)
4. [Docker](#docker)
5. [Contributing](#contributing)
6. [License](#license)



---

Feel free to replace `https://github.com/trudag/cv-specs-measurements.git` with the actual URL of your GitHub repository. This README covers the project structure, installation, usage, Docker integration, and contribution guidelines.
