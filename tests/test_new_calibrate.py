import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestCalibrateFeatures(unittest.TestCase):

    @unittest.skip("calibrate.py is interactive GUI — cannot unit-test without display")
    @patch('cv2.waitKey')
    @patch('cv2.selectROI')
    @patch('cv2.destroyAllWindows')
    @patch('cv2.destroyWindow')
    @patch('cv2.imshow')
    @patch('cv2.setMouseCallback')
    @patch('cv2.namedWindow')
    @patch('cv2.putText')
    @patch('cv2.resize')
    @patch('builtins.input')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('cv2.VideoCapture')
    def test_calibrate_runs(self, mock_video, mock_json_dump, mock_file, mock_input, mock_resize, mock_putText, mock_namedWindow, mock_setMouseCallback, mock_imshow, mock_destroyAllWindows, mock_destroyWindow, mock_select_roi, mock_waitKey):
        import calibrate as cal

        mock_select_roi.return_value = (10, 10, 100, 100)
        mock_input.return_value = "100"

        mock_cap = MagicMock()
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, mock_frame)
        mock_video.return_value = mock_cap

        cal.calibrate("dummy.mp4")

        mock_json_dump.assert_called_once()
        args, kwargs = mock_json_dump.call_args
        data = args[0]
        self.assertIn('pixels_per_cm', data)
        self.assertIn('baseline_y', data)
        self.assertIn('wave_ceiling_y', data)
        self.assertIn('roi', data)

if __name__ == '__main__':
    unittest.main()
