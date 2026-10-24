import cv2
from ultralytics import YOLO

# 1. Load the YOLO model
print("Loading model...")
model = YOLO('yolov8n.pt')

# 2. Open camera (0 is usually your built-in laptop webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam. If you don't have one, we can test with a video file instead!")
    exit()

print("Camera running! Tap the 'q' key while looking at the video window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection on the current frame
    results = model(frame, verbose=False)

    # Annotate the frame with bounding boxes and labels
    annotated_frame = results[0].plot()

    # Show the live feed
    cv2.imshow("Live AI Object Detector", annotated_frame)

    # Press 'q' on your keyboard to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()