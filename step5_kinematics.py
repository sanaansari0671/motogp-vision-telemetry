import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load YOLO Pose model
print("Loading model...")
model = YOLO('yolov8n-pose.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not access camera.")
    exit()

print("Lean Angle Estimator Running! Lean your torso left and right.")
print("Tap 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    # Check if a person is detected with keypoints
    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        # COCO Pose keypoint indices:
        # 5: Left Shoulder, 6: Right Shoulder
        # 11: Left Hip, 12: Right Hip
        kpts = results[0].keypoints.xy[0].cpu().numpy()

        if len(kpts) >= 13:
            l_shoulder, r_shoulder = kpts[5], kpts[6]
            l_hip, r_hip = kpts[11], kpts[12]

            # Calculate midpoints if all 4 keypoints are detected
            if np.all(l_shoulder > 0) and np.all(r_shoulder > 0) and np.all(l_hip > 0) and np.all(r_hip > 0):
                mid_shoulder = (l_shoulder + r_shoulder) / 2
                mid_hip = (l_hip + r_hip) / 2

                # Vector from Hip (pivot) to Shoulder (top)
                dx = mid_shoulder[0] - mid_hip[0]
                dy = mid_shoulder[1] - mid_hip[1]

                # Angle relative to vertical (dy is inverted in screen coordinates)
                angle_rad = np.arctan2(dx, -dy)
                angle_deg = np.degrees(angle_rad)

                # Draw spine vector line
                pt1 = (int(mid_hip[0]), int(mid_hip[1]))
                pt2 = (int(mid_shoulder[0]), int(mid_shoulder[1]))
                cv2.line(annotated_frame, pt1, pt2, (0, 255, 255), 4)

                # Display numerical angle on screen
                direction = "RIGHT" if angle_deg > 0 else "LEFT"
                abs_angle = abs(angle_deg)
                text = f"Torso Lean: {abs_angle:.1f} deg ({direction})"
                
                # Overlay background banner
                cv2.rectangle(annotated_frame, (20, 20), (450, 70), (0, 0, 0), -1)
                cv2.putText(
                    annotated_frame, 
                    text, 
                    (30, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (0, 255, 0), 
                    2
                )

    cv2.imshow("Real-Time Kinematics & Lean Angle Estimator", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()