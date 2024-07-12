import json
import cv2
import numpy as np
import unittest
from unittest.mock import MagicMock, call, mock_open, patch
from Supporting_code_depl.deplutilities import (
    order_points, angle_between_vectors,
    calculate_rectangle_angles, get_best_frames_from_json,
    display_frame, extract_and_return_frames, plot_quadrilateral
)

class TestFunctions(unittest.TestCase):

    def test_order_points(self):
        points = [(4, 5), (1, 2), (4, 2), (1, 5)]
        ordered = order_points(points)
        expected = [(1, 2), (4, 2), (4, 5), (1, 5)]
        self.assertEqual(ordered, expected, "The points should be ordered as top-left, top-right, bottom-right, bottom-left.")

    def test_angle_between_vectors(self):
        v1 = (1, 0)
        v2 = (0, 1)
        result = angle_between_vectors(v1, v2)
        expected = 90.0
        self.assertAlmostEqual(result, expected, places=5)

        v1 = (1, 0)
        v2 = (1, 0)
        result = angle_between_vectors(v1, v2)
        expected = 0.0
        self.assertAlmostEqual(result, expected, places=5)

        v1 = (1, 0)
        v2 = (-1, 0)
        result = angle_between_vectors(v1, v2)
        expected = 180.0
        self.assertAlmostEqual(result, expected, places=5)

    def test_calculate_rectangle_angles(self):
        points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        result = calculate_rectangle_angles(points)
        expected = [90.0, 90.0, 90.0, 90.0]
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e, places=5)

        points = [(0, 0), (2, 0), (2, 1), (0, 1)]
        result = calculate_rectangle_angles(points)
        expected = [90.0, 90.0, 90.0, 90.0]
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e, places=5)

        with self.assertRaises(ValueError):
            calculate_rectangle_angles([(0, 0), (1, 0), (1, 1)])

    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps([
        {
            "frame_number": 1,
            "coordinates": [(0, 0), (1, 0), (1, 1), (0, 1)]
        },
        {
            "frame_number": 2,
            "coordinates": [(0, 0), (2, 0), (2, 1), (0, 1)]
        },
        {
            "frame_number": 3,
            "coordinates": [(0, 0), (3, 0), (3, 1), (0, 1)]
        }
    ]))
    def test_get_best_frames_from_json(self, mock_file):
        result = get_best_frames_from_json('dummy_path', num_best_frames=2)
        expected_frame_numbers = [1, 2]
        self.assertEqual(len(result), 2)
        self.assertEqual([frame['frame_number'] for frame in result], expected_frame_numbers)

        result = get_best_frames_from_json('dummy_path', num_best_frames=1)
        expected_frame_numbers = [1]
        self.assertEqual(len(result), 1)
        self.assertEqual([frame['frame_number'] for frame in result], expected_frame_numbers)



    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.imshow')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.axis')
    @patch('cv2.cvtColor')
    def test_display_frame(self, mock_cvtColor, mock_axis, mock_title, mock_imshow, mock_show):
        frame_number = 1
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        display_frame(image, frame_number)

        mock_cvtColor.assert_called_once_with(image, cv2.COLOR_BGR2RGB)
        mock_imshow.assert_called_once_with(mock_cvtColor())
        mock_title.assert_called_once_with(f"Frame {frame_number}")
        mock_axis.assert_called_once_with('off')
        mock_show.assert_called_once()

        

if __name__ == '__main__':
    unittest.main()

