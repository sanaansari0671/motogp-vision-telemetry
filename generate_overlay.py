import cv2
import numpy as np
import pandas as pd
import json
import os

VIDEO_PATH = "motogp_turn.mp4"
DATA_PATH = "telemetry_data_v2.csv"
OUTPUT_PATH = "annotated_turn.mp4"

if not os.path.exists(VIDEO_PATH) or not os.path.exists(DATA_PATH):
    print("Required files missing! Make sure motogp_turn.mp4 and telemetry_data_v2.csv exist.")
    exit()

df = pd.read_csv(DATA_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

frame_idx = 0
print("Rendering live telemetry HUD overlay onto video frames...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    frame_data = df[df["frame"] == frame_idx]

    # Draw telemetry HUD overlay on screen
    cv2.rectangle(frame, (20, 20), (360, 140), (0, 0, 0), -1)
    cv2.rectangle(frame, (20, 20), (360, 140), (0, 255, 255), 2)
    cv2.putText(frame, "AI VISION TELEMETRY HUD", (35, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    y_offset = 75
    for _, row in frame_data.iterrows():
        comp = row["competitor"]
        spd = row["speed_kmh"]
        lean = abs(row["lean_angle_deg"])

        hud_text = f"{comp[:14]}: {spd:.0f} km/h | {lean:.0f} deg"
        cv2.putText(frame, hud_text, (35, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25

    out.write(frame)

cap.release()
out.release()
print(f"Annotated video saved successfully as {OUTPUT_PATH}!")