import cv2
from ultralytics import YOLO

# 1. Load the pre-trained YOLO Pose model (downloads 'yolov8n-pose.pt' automatically, ~6 MB)
print("Loading YOLOv8 Pose model...")
model = YOLO('yolov8n-pose.pt')

# 2. Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not access camera.")
    exit()

print("YOLO Pose tracking running! Stand back so your body is visible.")
print("Tap 'q' while focused on the video window to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run pose inference on the current frame
    results = model(frame, verbose=False)

    # Automatically draws the skeleton joints and limbs
    annotated_frame = results[0].plot()

    # Display the result
    cv2.imshow("AI Real-Time Pose Estimation (YOLO)", annotated_frame)

    # Press 'q' to close
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()