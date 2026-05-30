import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import process

class TestDetectSurface(unittest.TestCase):
    def test_detect_surface_exists(self):
        self.assertTrue(hasattr(process, 'detect_surface'))

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

    def test_detect_surface_ransac_slope(self):
        h, w = 100, 200
        frame = np.full((h, w, 3), 255, dtype=np.uint8)
        for col in range(w):
            surface_y = 60 + col * 20 // w
            frame[surface_y:, col, :] = 0
        cal = {
            'roi': {'x': 0, 'y': 0, 'w': w, 'h': h},
            'wave_ceiling_y': 5
        }
        surface_y, _, confidence = process.detect_surface(frame, cal)
        self.assertIsNotNone(surface_y)
        self.assertGreaterEqual(surface_y, 55)
        self.assertLessEqual(surface_y, 85)
        self.assertIn(confidence, ('primary', 'fallback'))

if __name__ == '__main__':
    unittest.main()
