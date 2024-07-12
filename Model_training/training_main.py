import torch
import os

from ultralytics import YOLO
from Supporting_code_train.trainutilities import is_file_in_folder
from Supporting_code_train.trainutilities import save_yaml_from_string

def main():
    # Specifying constants
    EPOCHS = 10
    IMAGE_SIZE = 256
    BATCH_SIZE = 16
    MODEL_NAME = 'toy_model'
    SAVE_DIR = './Model_training/runs'  # Directory where to save the runs

    # Project root directory
    CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))

    # Paths for YAML files
    MODEL_SPEC_PATH = os.path.join(CURRENT_FOLDER, 'model_spec.yaml')
    TRAINING_DATA_PATH = os.path.join(CURRENT_FOLDER, 'Sample_development_data\\datasets\\training_data')
    HOLDOUT_DATA_PATH = os.path.join(CURRENT_FOLDER, 'Sample_development_data\\datasets\\holdout_data')


    print('Specifying the model in ' + MODEL_SPEC_PATH)
    # Content for two_marker_classes.yaml
    example_yaml_string = """
    train: '""" + TRAINING_DATA_PATH + """'
    val: '""" + HOLDOUT_DATA_PATH +"""'
    nc: 2
    names: [0, 1]
    """

    print('Creating model specification .yaml file') 
    save_yaml_from_string(example_yaml_string, MODEL_SPEC_PATH)
   
    print('Initiating model training')

    # Check if GPU is available
    cuda_available = torch.cuda.is_available()
    print(f"Is CUDA available? {cuda_available}")

    if not cuda_available:
        print("CUDA is not available. Please check your CUDA installation and environment.")

    # Load the model configuration or weights (yolov8s.yaml is for two marker types)
    print('Loading an untrained YOLO model')
    model = YOLO('yolov8s.yaml')  # Specify the path to your pre-trained model weights or use 'yolov8s.pt' for pre-trained weights

    # Train the model
    print('Starting to train the model')
    model.train(data=MODEL_SPEC_PATH,
                epochs=EPOCHS,
                imgsz=IMAGE_SIZE,
                batch=BATCH_SIZE,
                project=SAVE_DIR,  # Specify the directory where to save runs
                name=MODEL_NAME)

if __name__ == '__main__':
    main()