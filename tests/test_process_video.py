import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from unittest.mock import MagicMock, patch
import process

class TestProcessVideo(unittest.TestCase):
    @patch('process.cv2.VideoCapture')
    @patch('process.load_cal')
    @patch('process.detect_surface')
    def test_process_video_two_pass(self, mock_detect, mock_load_cal, mock_video_capture):
        mock_cal = {
            'baseline_y': 500,
            'pixels_per_cm': 10.0,
            'fps': 30.0,
            'roi': {'x': 0, 'y': 0, 'w': 100, 'h': 100}
        }
        mock_load_cal.return_value = mock_cal

        fake_thresh = np.zeros((100, 100), dtype=np.uint8)

        mock_cap = MagicMock()
        mock_cap.read.side_effect = [
            (True, np.zeros((100, 100, 3), dtype=np.uint8)),
            (True, np.zeros((100, 100, 3), dtype=np.uint8)),
            (False, None)
        ]
        mock_cap.get.return_value = 100
        mock_video_capture.return_value = mock_cap

        mock_detect.side_effect = [
            (400, fake_thresh, 'primary'),
            (500, fake_thresh, 'fallback')
        ]

        results = process.process_video('dummy.mp4')

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['wave_height_cm'], 10.0)
        self.assertEqual(results[0]['detection_confidence'], 'primary')
        self.assertEqual(results[0]['frame_number'], 1)
        self.assertEqual(results[1]['wave_height_cm'], 0.0)
        self.assertEqual(results[1]['detection_confidence'], 'fallback')
        self.assertEqual(results[1]['frame_number'], 2)

if __name__ == '__main__':
    unittest.main()
