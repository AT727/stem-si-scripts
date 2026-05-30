import cv2
import numpy as np
import json
import os
from sklearn.linear_model import RANSACRegressor

def load_cal():
    with open("calibration.json", "r") as f:
        return json.load(f)

def detect_surface(frame, cal):
    MARGIN = 30
    
    roi = cal['roi']
    roi_view = frame[roi['y']:roi['y']+roi['h'], roi['x']:roi['x']+roi['w']]
    roi_width = roi['w']
    
    ceiling_y_local = cal['wave_ceiling_y'] - roi['y']
    
    gray = cv2.cvtColor(roi_view, cv2.COLOR_BGR2GRAY)
    masked = gray.copy()
    masked[:ceiling_y_local, :] = 255
    
    blurred = cv2.GaussianBlur(masked, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_width = roi['w']
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if h >= 8 and w >= roi_width * 0.3 and area >= 50 and y >= ceiling_y_local + 30:
            candidates.append((y, cnt))
    
    if not candidates:
        return None, thresh, 'uncertain'
    
    best_y, best_cnt = max(candidates, key=lambda c: cv2.contourArea(c[1]))
    
    envelope = {}
    for point in best_cnt:
        px, py = point[0]
        if px not in envelope or py < envelope[px]:
            envelope[px] = py
    
    env_for_ransac = {px: py for px, py in envelope.items() if py >= ceiling_y_local + 30}
    if len(env_for_ransac) < 10:
        env_for_ransac = envelope
    
    if len(env_for_ransac) >= 10:
        X = np.array(list(env_for_ransac.keys())).reshape(-1, 1)
        Y = np.array(list(env_for_ransac.values()))
        try:
            ransac = RANSACRegressor(
                min_samples=max(2, int(len(X) * 0.5)),
                residual_threshold=8.0
            )
            ransac.fit(X, Y)
            center_x = np.array([[roi_width // 2]])
            surface_local = int(ransac.predict(center_x)[0])
            inlier_ratio = np.sum(ransac.inlier_mask_) / len(X)
            if inlier_ratio >= 0.60:
                confidence = 'primary'
            elif inlier_ratio >= 0.35:
                confidence = 'fallback'
            else:
                confidence = 'uncertain'
        except ValueError:
            surface_local = int(np.median(Y))
            confidence = 'fallback'
    else:
        surface_local = best_y
        confidence = 'fallback'
    
    return surface_local + roi['y'], thresh, confidence

def process_video(video_path):
    cal = load_cal()
    cap = cv2.VideoCapture(video_path)
    fps = cal['fps']
    
    # Pass 1: Baseline
    baseline_y = cal['baseline_y']
    
    # Pass 2: Detection
    results = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_debug = 5
    debug_frames = {max(1, int(total_frames * i / (num_debug - 1))) for i in range(num_debug)}
    progress_interval = int(fps * 5)
    roi = cal['roi']
    
    os.makedirs("tmp", exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        
        is_debug = frame_count in debug_frames
        if is_debug:
            roi_debug = frame[roi['y']:roi['y']+roi['h'], roi['x']:roi['x']+roi['w']].copy()
        
        surface_y, thresh, confidence = detect_surface(frame, cal)
        if surface_y is None:
            surface_y = baseline_y
            confidence = 'uncertain'
        
        if confidence == 'uncertain':
            print(f"WARNING frame {frame_count}: low RANSAC inliers — result unreliable")
        
        pixel_displacement = baseline_y - surface_y
        wave_height_cm = round(max(0.0, pixel_displacement) / cal['pixels_per_cm'], 3)
        results.append({
            'frame_number': frame_count,
            'timestamp_s': round(frame_count / fps, 4),
            'wave_height_cm': wave_height_cm,
            'detection_confidence': confidence
        })
        
        if frame_count % progress_interval == 0:
            elapsed = frame_count / fps
            print(f"Processed {frame_count} frames ({elapsed:.1f}s)")
        
        if is_debug:
            surface_local = surface_y - roi['y']
            cv2.line(roi_debug, (0, surface_local), (roi['w']-1, surface_local), (0, 0, 255), 2)
            label = f"Frame {frame_count} | H = {wave_height_cm:.3f} cm"
            cv2.putText(roi_debug, label, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            debug_img = np.hstack((roi_debug, thresh_color))
            cv2.imwrite(f"tmp/debug_frame_{frame_count:04d}.png", debug_img)
    
    total_time = frame_count / fps
    print(f"Done. Processed {frame_count} frames in {total_time:.1f}s")
    
    return results
