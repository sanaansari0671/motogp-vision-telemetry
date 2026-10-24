import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# 1. Load YOLO model
model = YOLO('yolov8n.pt')

# 2. Track histories: store past centroid points for each object ID
track_history = defaultdict(lambda: [])

# 3. Source: We use a live test clip from online directly in code
# (You can also replace this URL with "my_clip.mp4" if you have a local video file)
video_url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4"

print("Opening video stream...")
cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print("Could not open video stream.")
    exit()

print("Tracking running! Vehicles are assigned unique IDs and trajectory tails.")
print("Tap 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO with built-in ByteTrack tracker
    # classes=[2, 3] detects cars (2) and motorcycles (3) from COCO dataset
    results = model.track(frame, persist=True, classes=[2, 3], verbose=False)

    if results[0].boxes and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            
            # Bottom-center footprint of the vehicle (contact point with the asphalt)
            footprint = (int(x), int(y + h / 2))

            # Store the trajectory point
            track = track_history[track_id]
            track.append(footprint)
            if len(track) > 30: # Keep the last 30 frames for the tail
                track.pop(0)

            # Draw bounding box
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Vehicle #{track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Draw the trailing trajectory line (racing line)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 255), thickness=3)

    cv2.imshow("Vehicle Trajectory & Kinematics Tracker", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()