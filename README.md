# Vision-Based MotoGP Kinematics & Competitor Telemetry Engine

An end-to-end computer vision pipeline that reconstructs real-world metric telemetry (velocity profiles, dynamic lean angle, and 2D ground trajectories) from broadcast race footage without onboard sensor data.

## System Pipeline

1. **Object Tracking:** Uses YOLOv8 with ByteTrack persistence to track competing riders through high-speed cornering battles.
2. **Homography Plane Rectification:** 4-point perspective transformation maps tyre contact patches from the camera plane onto an overhead metric $(X, Y)$ coordinate plane.
3. **Kinematics Engine:** Computes ground velocity ($\text{km/h}$) adjusted for slow-motion broadcast frame rates and estimates dynamic lean angle ($\theta$) from apex trajectory curvature.
4. **Interactive Telemetry Dashboard:** Streamlit application featuring synchronized video replay, rider KPI comparison cards, and bird's-eye racing line visualization.

## Tech Stack

- **Computer Vision:** OpenCV, Ultralytics YOLOv8, ByteTrack
- **Math & Data:** NumPy, Pandas
- **Dashboard & Plots:** Streamlit, Plotly Express
- **Video Encoding:** imageio, imageio-ffmpeg

## Quickstart

```powershell
# 1. Activate environment
.\venv\Scripts\Activate.ps1

# 2. Run telemetry extraction
python pipeline_engine_v2.py

# 3. Launch dashboard
streamlit run app.py