import cv2
import numpy as np
import json

with open("calibration.json") as f:
    cal = json.load(f)

roi = cal['roi']
baseline_y = cal['baseline_y']

for frame_num in [1, 500, 750, 1000]:
    cap = cv2.VideoCapture("PhaseII_TestD_0003_c4_01.MP4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        continue
    
    roi_view = frame[roi['y']:roi['y']+roi['h'], roi['x']:roi['x']+roi['w']]
    gray = cv2.cvtColor(roi_view, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    
    print(f"\n=== Frame {frame_num} ===")
    print(f"Baseline ROI y: {baseline_y - roi['y']}")
    
    # Check a few columns in detail
    for col in [50, 100, 150, 200]:
        col_grad = grad_y[:, col]
        # Find strongest negative and positive gradients
        pos_peaks = np.where(col_grad > 30)[0]
        neg_peaks = np.where(col_grad < -30)[0]
        abs_peaks = np.where(np.abs(col_grad) > 30)[0]
        
        print(f"  Col {col}:")
        print(f"    Positive (>30): {len(pos_peaks)} peaks, deepest at y={pos_peaks[-1] if len(pos_peaks) > 0 else 'none'}")
        print(f"    Negative (<-30): {len(neg_peaks)} peaks, deepest at y={neg_peaks[-1] if len(neg_peaks) > 0 else 'none'}")
        if len(abs_peaks) > 0:
            deepest = abs_peaks[-1]
            print(f"    Deepest abs>30: y={deepest}, grad={col_grad[deepest]:.0f}")
        
        # Check gradient at baseline region
        base_y_local = baseline_y - roi['y']
        for dy in [-5, 0, 5]:
            y = base_y_local + dy
            if 0 <= y < gray.shape[0]:
                print(f"    Gradient at baseline(dy={dy}): {col_grad[y]:.0f}")
        
        # Scan from bottom up, find first abs gradient > threshold
        for y in range(gray.shape[0] - 1, 0, -1):
            if abs(col_grad[y]) > 50:
                print(f"    Bottom-up first strong edge: y={y}, grad={col_grad[y]:.0f}")
                break
        
        # Show grayscale values around baseline
        print(f"    Grayscale at baseline-10:{gray[base_y_local-10, col]}, baseline:{gray[base_y_local, col]}, baseline+10:{gray[min(gray.shape[0]-1, base_y_local+10), col]}")
