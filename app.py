import os
import pandas as pd
import plotly.express as px
import streamlit as st

# Page configuration
st.set_page_config(page_title="MotoGP Telemetry v2", layout="wide")
st.title("🏁 Vision-Based MotoGP Telemetry & Kinematics")

DATA_FILE = "telemetry_data_v2.csv"

# Check if data exists
if not os.path.exists(DATA_FILE):
    st.error("telemetry_data_v2.csv not found! Run python pipeline_engine_v2.py first.")
    st.stop()

df = pd.read_csv(DATA_FILE)

# Sidebar Filter
st.sidebar.header("Competitor Selection")
available_competitors = df["competitor"].unique().tolist()
selected = st.sidebar.multiselect(
    "Select Competitors:",
    available_competitors,
    default=available_competitors[:2]
)

filtered_df = df[df["competitor"].isin(selected)]

# Video & KPI Metric Cards Section
st.subheader("Race Replay & Performance Metrics")
video_col, metric_col = st.columns(2)
with video_col:
    st.markdown("**Corner Replay (Turn 6 Apex)**")
    if os.path.exists("annotated_web.mp4"):
        st.video("annotated_web.mp4")
    elif os.path.exists("motogp_turn.mp4"):
        st.video("motogp_turn.mp4")
    elif os.path.exists("annotated_turn.mp4"):
        st.video("annotated_turn.mp4")
    else:
        st.info("No video file found.")

with metric_col:
    st.markdown("**Extracted Peak Metrics**")
    if selected:
        for comp in selected:
            c_data = filtered_df[filtered_df["competitor"] == comp]
            max_speed = c_data["speed_kmh"].max()
            max_lean = abs(c_data["lean_angle_deg"]).max()
            st.metric(f"{comp} Peak Speed", f"{max_speed:.1f} km/h")
            st.metric(f"{comp} Max Lean Angle", f"{max_lean:.1f}°")
    else:
        st.warning("Please select at least one competitor in the sidebar.")

st.markdown("---")

# Analytics Tabs
tab1, tab2 = st.tabs(["📈 Speed & Lean Dynamics", "🗺️ 2D Metric Racing Line"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        fig_speed = px.line(
            filtered_df,
            x="timestamp_sec",
            y="speed_kmh",
            color="competitor",
            title="Calibrated Velocity Profile (km/h vs Time)",
            labels={"timestamp_sec": "Time (s)", "speed_kmh": "Speed (km/h)"}
        )
        st.plotly_chart(fig_speed, use_container_width=True)

    with col_b:
        fig_lean = px.line(
            filtered_df,
            x="timestamp_sec",
            y="lean_angle_deg",
            color="competitor",
            title="Roll Angle Progression (° vs Time)",
            labels={"timestamp_sec": "Time (s)", "lean_angle_deg": "Lean Angle (°)"}
        )
        st.plotly_chart(fig_lean, use_container_width=True)

with tab2:
    fig_map = px.line(
        filtered_df,
        x="x_meter",
        y="y_meter",
        color="competitor",
        title="Bird's-Eye Overhead Racing Line (Turn 6 Apex - Metric Plane)",
        labels={"x_meter": "Track Width (m)", "y_meter": "Track Length (m)"}
    )
    fig_map.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_map, use_container_width=True)
