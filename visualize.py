import argparse
import json
import cv2
import pandas as pd
import numpy as np
from pathlib import Path

def generate_roi_png(video_path, cal, output_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret: return

    roi = cal["roi"]
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    
    # Draw ROI
    roi_img = frame.copy()
    cv2.rectangle(roi_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Draw baseline
    baseline_y = int(cal.get("baseline_y", y + h))
    cv2.line(roi_img, (x, baseline_y), (x+w, baseline_y), (0, 255, 255), 2)
    
    cv2.imwrite(str(output_path), roi_img)

def generate_annotated_mp4(video_path, cal, df, output_path, use_filtered=False):
    col = 'wave_height_filtered_cm' if use_filtered else 'wave_height_cm'
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    roi = cal["roi"]
    rx, ry, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]
    pixels_per_cm = cal["pixels_per_cm"]
    baseline_y = cal.get("baseline_y", ry + rh)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Get wave height for current frame
        row = df[df['frame_number'] == frame_idx]
        if not row.empty:
            wave_height = row.iloc[0][col]
            if not np.isnan(wave_height):
                # Convert cm back to pixel Y
                pixel_displacement = wave_height * pixels_per_cm
                surface_y = int(baseline_y - pixel_displacement)
                
                # Draw red dot
                cv2.circle(frame, (rx + 10, surface_y), 5, (0, 0, 255), -1)
            
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()

def main():
    parser = argparse.ArgumentParser(description="Visualize detection pipeline.")
    parser.add_argument("video", help="Input video file")
    parser.add_argument("--roi-png", action="store_true", help="Generate ROI verification PNG")
    parser.add_argument("--raw-mp4", action="store_true", help="Generate raw detection MP4")
    parser.add_argument("--filtered-mp4", action="store_true", help="Generate filtered detection MP4")
    args = parser.parse_args()

    # Load calibration and results
    with open("calibration.json") as f:
        cal = json.load(f)
    df = pd.read_csv("output.csv")
    
    if args.roi_png:
        generate_roi_png(args.video, cal, "roi_verification.png")
        print("Generated roi_verification.png")
    
    if args.raw_mp4:
        generate_annotated_mp4(args.video, cal, df, "annotated_raw.mp4", use_filtered=False)
        print("Generated annotated_raw.mp4")
        
    if args.filtered_mp4:
        generate_annotated_mp4(args.video, cal, df, "annotated_filtered.mp4", use_filtered=True)
        print("Generated annotated_filtered.mp4")

if __name__ == "__main__":
    main()
