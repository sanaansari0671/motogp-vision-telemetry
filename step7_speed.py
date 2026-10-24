import cv2
import numpy as np
import time
from collections import defaultdict
from ultralytics import YOLO

# 1. Load YOLO model
model = YOLO('yolov8n.pt')

# 2. Source video
video_url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4"
cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print("Could not open video stream.")
    exit()

# Get video FPS to measure delta time (dt) accurately
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps):
    fps = 30.0
time_per_frame = 1.0 / fps

# 3. Define Homography Reference Points
# In the camera frame, a stretch of highway lane forms a trapezoid (pixel coordinates)
# [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
src_pts = np.float32([
    [200, 260],   # Far left lane point
    [580, 260],   # Far right lane point
    [760, 560],   # Near right lane point
    [50,  560]    # Near left lane point
])

# Real-world metric dimensions corresponding to that trapezoid:
# Say, 15 meters wide by 40 meters long
REAL_WIDTH_METERS = 15.0
REAL_LENGTH_METERS = 40.0

# Destination canvas in meters scaled to pixels (10 pixels = 1 meter)
CANVAS_SCALE = 10 
dst_pts = np.float32([
    [0, 0],
    [REAL_WIDTH_METERS * CANVAS_SCALE, 0],
    [REAL_WIDTH_METERS * CANVAS_SCALE, REAL_LENGTH_METERS * CANVAS_SCALE],
    [0, REAL_LENGTH_METERS * CANVAS_SCALE]
])

# Compute the Homography Transform Matrix (H)
H = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Track histories: stores past metric positions (x_meter, y_meter) for speed calculation
metric_history = defaultdict(lambda: [])
speeds_kmh = {}

print("Speed & Telemetry Estimator running! Watch vehicles get tracked with real km/h.")
print("Tap 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Track cars (class 2)
    results = model.track(frame, persist=True, classes=[2], verbose=False)

    # Draw the calibrated measurement zone polygon (cyan) on the road
    pts_zone = src_pts.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts_zone], isClosed=True, color=(255, 255, 0), thickness=2)
    cv2.putText(frame, "CALIBRATED TELEMETRY ZONE", (210, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if results[0].boxes and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            
            # Bottom-center footprint of the vehicle touching the road
            footprint_px = np.array([[[float(x), float(y + h / 2)]]], dtype=np.float32)

            # Transform pixel coordinate into bird's-eye 2D space using Homography
            transformed_pt = cv2.perspectiveTransform(footprint_px, H)[0][0]
            
            # Convert canvas pixels to actual meters
            x_meter = transformed_pt[0] / CANVAS_SCALE
            y_meter = transformed_pt[1] / CANVAS_SCALE

            history = metric_history[track_id]
            history.append((x_meter, y_meter))
            if len(history) > 10:
                history.pop(0)

            # Compute speed if we have at least 5 frames of position history
            if len(history) >= 5:
                start_x, start_y = history[0]
                curr_x, curr_y = history[-1]
                
                # Euclidean distance traveled in meters
                dist_meters = np.sqrt((curr_x - start_x)**2 + (curr_y - start_y)**2)
                dt = (len(history) - 1) * time_per_frame
                
                # Speed = meters/second -> converted to km/h (multiply by 3.6)
                speed = (dist_meters / dt) * 3.6
                
                # Apply an exponential moving average for smooth readout
                if track_id in speeds_kmh:
                    speeds_kmh[track_id] = 0.7 * speeds_kmh[track_id] + 0.3 * speed
                else:
                    speeds_kmh[track_id] = speed

            # Draw vehicle bounding box and speed tag
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            current_speed = speeds_kmh.get(track_id, 0.0)
            label = f"ID #{track_id} | {current_speed:.1f} km/h"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Vision-Based Velocity & Telemetry Estimator", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()