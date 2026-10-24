import cv2
import json

pts = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append([x, y])
        cv2.circle(frame_display, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            frame_display, 
            f"P{len(pts)}: ({x},{y})", 
            (x + 10, y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 255), 
            1
        )
        cv2.imshow("Track Calibration - Click 4 Corners", frame_display)
        if len(pts) == 4:
            print(f"\nCaptured 4 points: {pts}")
            with open("track_config.json", "w") as f:
                json.dump(pts, f)
            print("Saved coordinates to track_config.json! Tap any key on the video window to close it.")

# Open our sliced video
cap = cv2.VideoCapture("motogp_turn.mp4")
ret, frame = cap.read()
cap.release()

if not ret:
    print("Could not read motogp_turn.mp4. Make sure the file exists in this directory!")
    exit()

frame_display = frame.copy()
cv2.imshow("Track Calibration - Click 4 Corners", frame_display)
print("Click 4 points on the track surface in order:")
print("1. Far Top-Left  |  2. Far Top-Right  |  3. Near Bottom-Right  |  4. Near Bottom-Left")

cv2.setMouseCallback("Track Calibration - Click 4 Corners", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()