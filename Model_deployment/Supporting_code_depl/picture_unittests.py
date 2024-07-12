import unittest
import cv2
import numpy as np
from unittest.mock import patch, MagicMock
from Supporting_code_depl.deplutilities import show_image, detect_face, crop_face, map_coordinates, draw_circles, process_image

class TestImageProcessing(unittest.TestCase):
    def setUp(self):
        self.image_path = 'Model_deployment/Supporting_code_depl/Unit_test_data/53986.png'
        self.image = cv2.imread(self.image_path)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.marker_detections = [
            [0, 0.5, 0.5, 0.1, 0.1],
            [1, 0.3, 0.3, 0.05, 0.05]
        ]

    def test_show_image(self):
        with patch('matplotlib.pyplot.show') as mock_show:
            show_image(self.image, title='Test Image')
            mock_show.assert_called_once()

    def test_detect_face(self):
        face_rect, image_with_faces = detect_face(self.image)
        self.assertIsNotNone(face_rect, "Face should be detected.")
        self.assertEqual(len(face_rect), 4, "Detected face rectangle should have 4 elements (x, y, w, h).")

    def test_crop_face(self):
        face_rect = (200, 200, 100, 100)
        cropped_image, offset = crop_face(self.image, face_rect, padding=1.0)
        expected_offset = (100, 100)
        self.assertEqual(offset, expected_offset, "Offset should match the expected value.")
        self.assertEqual(cropped_image.shape, (300, 300, 3), "Cropped image shape should match the expected dimensions.")

    def test_map_coordinates(self):
        offset = (100, 100)
        cropped_shape = (300, 300)
        original_shape = self.image.shape[:2]
        mapped_detections = map_coordinates(self.marker_detections, offset, cropped_shape, original_shape)
        self.assertEqual(len(mapped_detections), 2, "Two markers should be mapped.")
        self.assertEqual(mapped_detections[0]['x'], 250, "Mapped x-coordinate should match the expected value.")
        self.assertEqual(mapped_detections[0]['y'], 250, "Mapped y-coordinate should match the expected value.")

    def test_draw_circles(self):
        detections = [
            {'marker': 0, 'x': 250, 'y': 250, 'width': 50, 'height': 50},
            {'marker': 1, 'x': 150, 'y': 150, 'width': 25, 'height': 25}
        ]
        draw_circles(self.image, detections)
        self.assertTrue(np.any(self.image != np.zeros(self.image.shape, dtype=np.uint8)), "Image should have circles drawn on it.")

if __name__ == '__main__':
    unittest.main()


