import cv2
import numpy as np
import pandas as pd
import json
import os
from collections import defaultdict
from ultralytics import YOLO

CONFIG_PATH = "track_config.json"
VIDEO_PATH = "motogp_turn.mp4"

if not os.path.exists(CONFIG_PATH):
    print(f"Error: {CONFIG_PATH} not found. Run python calibrate_track.py first!")
    exit()

if not os.path.exists(VIDEO_PATH):
    print(f"Error: {VIDEO_PATH} not found!")
    exit()

# 1. Load your calibrated 4 corners
with open(CONFIG_PATH, "r") as f:
    src_pts = np.float32(json.load(f))

# Spielberg corner segment scale (approx 14m track width x 30m length)
REAL_WIDTH_METERS = 14.0
REAL_LENGTH_METERS = 30.0
CANVAS_SCALE = 10

dst_pts = np.float32([
    [0, 0],
    [REAL_WIDTH_METERS * CANVAS_SCALE, 0],
    [REAL_WIDTH_METERS * CANVAS_SCALE, REAL_LENGTH_METERS * CANVAS_SCALE],
    [0, REAL_LENGTH_METERS * CANVAS_SCALE]
])

H = cv2.getPerspectiveTransform(src_pts, dst_pts)

# 2. Load YOLO and video
print("Loading YOLOv8 tracking model...")
detector = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
time_per_frame = 1.0 / fps

metric_history = defaultdict(lambda: [])
speeds_kmh = {}
records = []
frame_idx = 0

print("Running pipeline on motogp_turn.mp4... Tracking bikes and extracting telemetry.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    # COCO Class 3 = motorcycle
    results = detector.track(frame, persist=True, classes=[3], verbose=False)

    if results[0].boxes and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            
            # Contact point of the bike tyres on the asphalt
            footprint_px = np.array([[[float(x), float(y + h / 2)]]], dtype=np.float32)

            # Convert perspective pixel to metric ground plane
            transformed_pt = cv2.perspectiveTransform(footprint_px, H)[0][0]
            x_m = float(transformed_pt[0] / CANVAS_SCALE)
            y_m = float(transformed_pt[1] / CANVAS_SCALE)

            history = metric_history[track_id]
            history.append((x_m, y_m))
            if len(history) > 10:
                history.pop(0)

            speed = 0.0
            if len(history) >= 5:
                sx, sy = history[0]
                dist = np.sqrt((x_m - sx)**2 + (y_m - sy)**2)
                dt = (len(history) - 1) * time_per_frame
                raw_speed = (dist / dt) * 3.6
                
                if track_id in speeds_kmh:
                    speed = 0.7 * speeds_kmh[track_id] + 0.3 * raw_speed
                else:
                    speed = raw_speed
                speeds_kmh[track_id] = speed

            # Estimate lean angle from trajectory delta
            lean_angle = 0.0
            if len(history) >= 3:
                dx = history[-1][0] - history[-3][0]
                dy = history[-1][1] - history[-3][1]
                curvature = dx / (abs(dy) + 1e-5)
                lean_angle = float(np.clip(curvature * 25.0, -65.0, 65.0))

            records.append({
                "frame": frame_idx,
                "timestamp_sec": round(frame_idx * time_per_frame, 2),
                "track_id": track_id,
                "x_meter": round(x_m, 2),
                "y_meter": round(y_m, 2),
                "speed_kmh": round(speed, 2),
                "lean_angle_deg": round(lean_angle, 2)
            })

cap.release()

# 3. Save to CSV
df = pd.DataFrame(records)
df.to_csv("telemetry_data.csv", index=False)
print(f"\nDone! Processed {frame_idx} frames and saved {len(df)} telemetry rows to telemetry_data.csv")