import pytest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import process

def test_imports():
    # Verify imports inside process
    assert 'json' in sys.modules
    assert 'cv2' in sys.modules
    assert 'sklearn' in sys.modules

def test_functions_exist():
    assert hasattr(process, 'load_cal')
    assert hasattr(process, 'process_video')
