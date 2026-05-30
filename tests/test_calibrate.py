import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestCalibrate(unittest.TestCase):
    @patch('cv2.VideoCapture')
    @patch('builtins.print')
    def test_calibrate_video_read_fails(self, mock_print, mock_video):
        import calibrate as cal
        mock_cap = MagicMock()
        mock_cap.read.return_value = (False, None)
        mock_video.return_value = mock_cap

        cal.calibrate("invalid.mp4")
        mock_print.assert_called_with("Error: Could not read video frame.")
