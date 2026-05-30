import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import process

class TestDetectSurface(unittest.TestCase):
    def test_detect_surface_exists(self):
        self.assertTrue(hasattr(process, 'detect_surface'), "detect_surface function should exist in process module")

    def test_detect_surface_basic(self):
        h, w = 100, 100
        frame = np.full((h, w, 3), 255, dtype=np.uint8)
        frame[50:, :, :] = 0

        cal = {
            'roi': {'x': 0, 'y': 0, 'w': w, 'h': h},
            'wave_ceiling_y': 5
        }

        surface_y, _, confidence = process.detect_surface(frame, cal)
        self.assertIsNotNone(surface_y)
        self.assertIn(confidence, ('primary', 'fallback', 'uncertain'))

if __name__ == '__main__':
    unittest.main()
