import unittest
import sys
import os
import argparse
from ultralytics import YOLO

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from Supporting_code_depl.deplutilities import process_image

# Add argument parsing
parser = argparse.ArgumentParser(description='Picture analysis script')
parser.add_argument('--model', type=str, required=True, help='Path to the model weights')
parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output images')
args = parser.parse_args()

# Declaring the constants from parsed arguments
MODEL_NAME = args.model
INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir


# Ensure the output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Performing unit tests
# Load the test suite from the test_image_processing module
print('Performing unit testing')
loader = unittest.TestLoader()
suite = loader.discover(start_dir='./Model_deployment/Supporting_code_depl', pattern='picture_unittests.py')

# Run the test suite
runner = unittest.TextTestRunner()
runner.run(suite)

# Load the model
print('Loading the model')
model = YOLO(MODEL_NAME)

# Check for valid files in the input directory
valid_files = [f for f in os.listdir(INPUT_DIR) if os.path.splitext(f)[1] in ['.png', '.jpg']]
if not valid_files:
    print('No valid .jpg or .png files found in the input directory.')
    sys.exit(1)

# Iterate over all valid files in the input directory
print('Processing images')
for filename in valid_files:
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, 'output_' + filename)
    print(f'Processing {input_path}')
    process_image(model=model,
                  image_path=input_path,
                  output_path=output_path,
                  face_crop_padding=0.8,
                  apply_box=False)

print('Done!')