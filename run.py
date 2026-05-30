from process import process_video
import csv

results = process_video("PhaseII_TestD_0003_c4_01.MP4")

with open("results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["frame_number", "timestamp_s", "wave_height_cm", "detection_confidence"])
    writer.writeheader()
    writer.writerows(results)
