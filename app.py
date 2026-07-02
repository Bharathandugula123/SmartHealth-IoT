from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parent
STREAM_PATH = ROOT / "data/generated/virtual_patient_streams.csv"
ALERT_PATH = ROOT / "data/results/hybrid_alert_log.csv"
SUMMARY_PATH = ROOT / "data/results/classification_results_summary.csv"

st.set_page_config(page_title="SmartHealth-IoT Dashboard", layout="wide")
st.title("SmartHealth-IoT Remote Patient Monitoring Dashboard")
st.caption("Simulation-based virtual wearable physiological stream monitoring. Not for clinical use.")

if not STREAM_PATH.exists():
    st.warning("Run `python main.py` first to generate simulated data and results.")
    st.stop()

df = pd.read_csv(STREAM_PATH)
patients = sorted(df.patient_id.unique())
patient = st.sidebar.selectbox("Patient", patients)
g = df[df.patient_id == patient].copy().reset_index(drop=True)
latest = g.iloc[-1]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Heart Rate", f"{latest.heart_rate:.1f} bpm")
c2.metric("SpO₂", f"{latest.spo2:.1f}%")
c3.metric("Temperature", f"{latest.temperature:.2f} °C")
c4.metric("Activity", f"{latest.activity:.1f}")
c5.metric("Risk Status", latest.true_label)

trend = g.tail(120).melt(id_vars=["timestamp"], value_vars=["heart_rate", "spo2", "temperature", "activity"], var_name="Signal", value_name="Value")
st.plotly_chart(px.line(trend, x="timestamp", y="Value", color="Signal", title="Recent Physiological Trends"), use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Patient Stream")
    st.dataframe(g.tail(25), use_container_width=True)
with right:
    st.subheader("Model Summary")
    if SUMMARY_PATH.exists():
        st.dataframe(pd.read_csv(SUMMARY_PATH), use_container_width=True)
    else:
        st.info("No model summary yet.")

if ALERT_PATH.exists():
    alerts = pd.read_csv(ALERT_PATH)
    pa = alerts[alerts.patient_id == patient].tail(20)
    st.subheader("Recent Alerts")
    st.dataframe(pa, use_container_width=True)
