import unittest
import sys
import os
import argparse
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from Supporting_code_depl.deplutilities import (
    order_points, angle_between_vectors, process_video,
    calculate_rectangle_angles, get_best_frames_from_json,
    visualize_video_with_detections,
    plot_quadrilateral, display_frame, extract_and_return_frames
)

# Add argument parsing
parser = argparse.ArgumentParser(description='Video analysis script')
parser.add_argument('--model', type=str, required=True, help='Path to the model weights')
parser.add_argument('--input_video', type=str, required=True, help='Path to the input video')
parser.add_argument('--output_video', type=str, required=True, help='Path to save the output video with detections')
parser.add_argument('--output_json', type=str, required=True, help='Path to save the output JSON results')
parser.add_argument('--process_video', action='store_true', help='Flag to process the video')
parser.add_argument('--show_video', action='store_true', help='Flag to show the video with detections')
args = parser.parse_args()

# Declaring the constants from parsed arguments
MODEL_NAME = args.model
INPUT_VIDEO_PATH = args.input_video
OUTPUT_VIDEO_PATH = args.output_video
OUTPUT_JSON_PATH = args.output_json
PROCESS_VIDEO = args.process_video
SHOW_VIDEO = args.show_video

# Performing unit tests
# Load the test suite from the test_image_processing module
print('Performing unit testing')
loader = unittest.TestLoader()
suite = loader.discover(start_dir='./Model_deployment/Supporting_code_depl', pattern='video_unittests.py')

# Run the test suite
runner = unittest.TextTestRunner()
runner.run(suite)

# Initiating a model
model = YOLO(MODEL_NAME)

# Example usage
if PROCESS_VIDEO:
    process_video(model, INPUT_VIDEO_PATH, OUTPUT_JSON_PATH)

if SHOW_VIDEO:
    visualize_video_with_detections(model, INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH)

# Example usage
best_frames = get_best_frames_from_json(OUTPUT_JSON_PATH, num_best_frames=10)
for frame in best_frames:
    print(frame)

# To get the very best frame, you can call the function with num_best_frames=1
best_frame = get_best_frames_from_json(OUTPUT_JSON_PATH, num_best_frames=1)
print("Best Frame:", best_frame[0])

# Outputting the best frame
frame_numbers = [best_frame[0]['frame_number']]  # List of frame numbers to extract
frames = extract_and_return_frames(INPUT_VIDEO_PATH, os.path.dirname(OUTPUT_VIDEO_PATH), frame_numbers)

# Save the best frame information to a text file
best_frame_path = os.path.join(os.path.dirname(OUTPUT_VIDEO_PATH), 'best_frame.txt')
with open(best_frame_path, 'w') as f:
    f.write(f"Best Frame Number: {best_frame[0]['frame_number']}\n")
    f.write(f"Coordinates: {best_frame[0]['coordinates']}\n")
print(f"Best frame information saved to {best_frame_path}")

# Display the extracted frames
for frame_number, image in frames.items():
    display_frame(image, frame_number)

# Plot all quadrilaterals
# plt.figure(figsize=(10, 10))
# for frame in best_frames:
#     plot_quadrilateral(frame)
# plt.gca().invert_yaxis()
# plt.gca().set_aspect('equal', adjustable='box')
# plt.title("Quadrilaterals from Frames")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()
