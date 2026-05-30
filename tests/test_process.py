import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class TestProcess(unittest.TestCase):
    def test_imports(self):
        import process
        self.assertTrue(hasattr(process, 'load_cal'))
        self.assertTrue(hasattr(process, 'process_video'))
        self.assertTrue(hasattr(process, 'detect_surface'))

    def test_functions_exist(self):
        import process
        self.assertTrue(callable(process.load_cal))
        self.assertTrue(callable(process.process_video))
        self.assertTrue(callable(process.detect_surface))

if __name__ == '__main__':
    unittest.main()
