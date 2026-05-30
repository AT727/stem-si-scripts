import unittest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from unittest.mock import MagicMock, patch
import process

class TestProcessVideo(unittest.TestCase):
    @patch('process.cv2.VideoCapture')
    @patch('process.load_cal')
    @patch('process.detect_surface')
    def test_process_video_two_pass(self, mock_detect, mock_load_cal, mock_video_capture):
        # Setup
        mock_cal = {
            'baseline_y': 500,
            'pixels_per_cm': 10.0
        }
        mock_load_cal.return_value = mock_cal
        
        mock_cap = MagicMock()
        # Simulate 2 frames
        mock_cap.read.side_effect = [
            (True, np.zeros((100, 100, 3))),
            (True, np.zeros((100, 100, 3))),
            (False, None)
        ]
        mock_video_capture.return_value = mock_cap
        
        # Simulate detect_surface returns
        mock_detect.side_effect = [400, 500] # surface_y
        
        # Run
        results = process.process_video('dummy.mp4')
        
        # Assert
        # Frame 1: pixel_displacement = 500 - 400 = 100. wave_height = 100 / 10 = 10.0
        # Frame 2: pixel_displacement = 500 - 500 = 0. wave_height = 0 / 10 = 0.0
        self.assertEqual(results, [10.0, 0.0])
        self.assertEqual(mock_cap.read.call_count, 3)

if __name__ == '__main__':
    unittest.main()
