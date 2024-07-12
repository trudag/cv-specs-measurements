import unittest
from unittest.mock import patch
import cv2
import numpy as np
import os


from Supporting_code_aug.augutilities import (
    create_dataset_from_images,
    apply_random_rotation,
    add_markers_to_image,
    apply_random_image_rotation
)


class TestImageProcessing(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((500, 500, 3), dtype=np.uint8)  # Black square image
        self.marker = np.ones((50, 50, 3), dtype=np.uint8) * 255  # White square marker
        self.markers_folder = './Data_augmentation/Supporting_code_aug/Unit_test_data/test_markers/'
        os.makedirs(self.markers_folder, exist_ok=True)
        cv2.imwrite(os.path.join(self.markers_folder, 'marker.png'), self.marker)
        
        self.output_folder = './Data_augmentation/Supporting_code_aug/Unit_test_data/test_output/'
        os.makedirs(self.output_folder, exist_ok=True)
        self.background_folder = './Data_augmentation/Supporting_code_aug/Unit_test_data/test_backgrounds/'
        os.makedirs(self.background_folder, exist_ok=True)
        cv2.imwrite(os.path.join(self.background_folder, 'background.png'), self.image)
        
        self.round_markers = [self.marker]
        self.square_markers = [self.marker]
    
    def test_apply_random_rotation(self):
        rotated_image = apply_random_rotation(self.image, apply_rotation=True)
        self.assertEqual(rotated_image.shape[2], 4, "Image should have 4 channels (RGBA) after rotation.")

    def test_add_markers_to_image(self):
        augmented_image, marker_positions = add_markers_to_image(self.image, self.round_markers, self.square_markers, random_marker_rotation=True)
        self.assertTrue(len(marker_positions) > 0, "Markers should be added to the image.")
        self.assertEqual(augmented_image.shape, self.image.shape, "Augmented image should have the same shape as the background image.")

    def test_apply_random_image_rotation(self):
        rotated_image = apply_random_image_rotation(self.image)
        self.assertEqual(rotated_image.shape, self.image.shape, "Rotated image should have the same shape as the original image.")

    def test_create_dataset_from_images(self):
        dataset_info = create_dataset_from_images(
            n_total_images=5, num_images_per_bg=1, num_background_images=2,
            round_markers=self.round_markers, square_markers=self.square_markers,
            background_folder=self.background_folder, output_folder=self.output_folder,
            random_marker_rotation=True, random_image_rotation=True,
            convert_to_bw=False, apply_edge_detection=False, apply_brightness_shift=True
        )
        self.assertTrue(len(dataset_info) > 0, "Dataset should be created with information about the images.")
        self.assertTrue(os.path.exists(self.output_folder), "Output folder should be created with images and labels.")

if __name__ == '__main__':
    unittest.main()
