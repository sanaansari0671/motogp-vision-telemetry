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
    print(f"Error: {CONFIG_PATH} not found!")
    exit()

if not os.path.exists(VIDEO_PATH):
    print(f"Error: {VIDEO_PATH} not found!")
    exit()

# 1. Load Calibration Corners
with open(CONFIG_PATH, "r") as f:
    src_pts = np.float32(json.load(f))

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

# 2. Setup YOLO with ByteTrack config
print("Loading YOLOv8 tracking model...")
detector = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
time_per_frame = 1.0 / fps

# Slow-mo broadcast compensation factor (replay is ~3x to 4x slower than real time)
SLOW_MO_FACTOR = 3.5 

metric_history = defaultdict(lambda: [])
speeds_kmh = {}
records = []
frame_idx = 0

print("Processing video with enhanced tracking & smoothing...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    
    # Track motorcycles (COCO class 3) with higher persistence
    results = detector.track(
        frame, 
        persist=True, 
        classes=[3], 
        tracker="bytetrack.yaml", 
        verbose=False
    )

    if results[0].boxes and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            
            # Bottom contact patch of the rear tire
            footprint_px = np.array([[[float(x), float(y + h / 2)]]], dtype=np.float32)

            transformed_pt = cv2.perspectiveTransform(footprint_px, H)[0][0]
            x_m = float(transformed_pt[0] / CANVAS_SCALE)
            y_m = float(transformed_pt[1] / CANVAS_SCALE)

            # Ignore points projected far outside the track boundaries
            if not (-5.0 <= x_m <= REAL_WIDTH_METERS + 5.0 and -5.0 <= y_m <= REAL_LENGTH_METERS + 5.0):
                continue

            history = metric_history[track_id]
            history.append((x_m, y_m))
            if len(history) > 12:
                history.pop(0)

            speed = 0.0
            if len(history) >= 4:
                # Rolling window distance
                dx = history[-1][0] - history[0][0]
                dy = history[-1][1] - history[0][1]
                dist = np.sqrt(dx**2 + dy**2)
                dt = (len(history) - 1) * time_per_frame
                
                # Raw speed adjusted for broadcast slow-mo factor
                raw_speed = (dist / dt) * 3.6 * SLOW_MO_FACTOR
                
                # Smoothing
                if track_id in speeds_kmh:
                    speed = 0.8 * speeds_kmh[track_id] + 0.2 * raw_speed
                else:
                    speed = raw_speed
                speeds_kmh[track_id] = speed

            # Curvature-derived roll angle
            lean_angle = 0.0
            if len(history) >= 4:
                dx = history[-1][0] - history[-3][0]
                dy = history[-1][1] - history[-3][1]
                curvature = dx / (abs(dy) + 1e-5)
                lean_angle = float(np.clip(curvature * 22.0, -65.0, 65.0))

            records.append({
                "frame": frame_idx,
                "timestamp_sec": round(frame_idx * time_per_frame, 2),
                "track_id": track_id,
                "x_meter": round(x_m, 2),
                "y_meter": round(y_m, 2),
                "speed_kmh": round(speed, 1),
                "lean_angle_deg": round(lean_angle, 1)
            })

cap.release()

df = pd.DataFrame(records)

# Filter out transient false-positive detections (must appear in at least 15 frames)
valid_ids = df['track_id'].value_counts()
valid_ids = valid_ids[valid_ids >= 15].index
df = df[df['track_id'].isin(valid_ids)]

# Remap dominant tracks to clear competitor labels
dominant_ids = df['track_id'].value_counts().head(2).index.tolist()
label_map = {}
if len(dominant_ids) >= 1:
    label_map[dominant_ids[0]] = "Competitor #89 (Martin)"
if len(dominant_ids) >= 2:
    label_map[dominant_ids[1]] = "Competitor #37 (Acosta)"

df['competitor'] = df['track_id'].map(lambda x: label_map.get(x, f"Rider #{x}"))

df.to_csv("telemetry_data_v2.csv", index=False)
print(f"Extraction complete! Filtered and saved {len(df)} telemetry rows to telemetry_data_v2.csv")