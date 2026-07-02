from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import label_binarize
from .utils import project_root, ID_TO_LABEL

FIG_DPI = 600

def _save(path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()

def plot_sample_stream(df: pd.DataFrame, out="figures/sample_physiological_stream.png"):
    pid = df["patient_id"].iloc[0]
    g = df[df.patient_id == pid].copy().reset_index(drop=True)
    plt.figure(figsize=(10, 4))
    plt.plot(g.index, g["heart_rate"], label="Heart Rate")
    plt.plot(g.index, g["spo2"], label="SpO₂")
    plt.plot(g.index, g["temperature"] * 2.5, label="Temperature ×2.5")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title(f"Virtual Wearable Physiological Stream ({pid})")
    plt.legend()
    _save(project_root() / out)

def plot_model_comparison(results: pd.DataFrame, out="figures/model_comparison_macro_f1.png"):
    metric = "macro_f1_mean" if "macro_f1_mean" in results.columns else "macro_f1"
    data = results.sort_values(metric, ascending=False)
    plt.figure(figsize=(9, 4))
    plt.bar(data["model"], data[metric])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Macro-F1")
    plt.title("Model Comparison for SmartHealth-IoT Risk Classification")
    _save(project_root() / out)

def plot_confusion(y_true, y_pred, out="figures/confusion_matrix_vitalboostnet.png"):
    plt.figure(figsize=(5, 4))
    ax = plt.gca()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=[0,1,2], display_labels=[ID_TO_LABEL[i] for i in range(3)], ax=ax, values_format="d")
    plt.title("VitalBoostNet Confusion Matrix")
    _save(project_root() / out)

def plot_multiclass_roc(y_true, proba, out="figures/roc_auc_vitalboostnet.png"):
    y_bin = label_binarize(y_true, classes=[0,1,2])
    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    for i in range(3):
        try:
            RocCurveDisplay.from_predictions(y_bin[:, i], proba[:, i], name=ID_TO_LABEL[i], ax=ax)
        except Exception:
            pass
    plt.title("VitalBoostNet Multiclass ROC Curves")
    _save(project_root() / out)

def plot_alert_metrics(alert_metrics: dict, out="figures/alert_performance.png"):
    keys = ["event_sensitivity", "false_alert_rate", "missed_event_rate"]
    vals = [alert_metrics.get(k, 0) for k in keys]
    plt.figure(figsize=(6, 4))
    plt.bar([k.replace("_", " ").title() for k in keys], vals)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Hybrid Alert Evaluation")
    _save(project_root() / out)

def plot_scalability(df: pd.DataFrame, out_latency="figures/latency_scalability.png", out_throughput="figures/throughput_scalability.png"):
    plt.figure(figsize=(7, 4))
    plt.plot(df["patient_load"], df["avg_ingestion_latency_ms"], marker="o")
    plt.xlabel("Virtual Patient Load")
    plt.ylabel("Average Ingestion Latency (ms)")
    plt.title("Latency Scalability Under Increasing Virtual Patient Load")
    _save(project_root() / out_latency)
    plt.figure(figsize=(7, 4))
    plt.plot(df["patient_load"], df["throughput_packets_per_second"], marker="o")
    plt.xlabel("Virtual Patient Load")
    plt.ylabel("Throughput (Packets/s)")
    plt.title("Throughput Scalability Under Increasing Virtual Patient Load")
    _save(project_root() / out_throughput)

def make_dashboard_snapshot(df: pd.DataFrame, alerts: pd.DataFrame | None = None, out="figures/dashboard_snapshot.png"):
    pid = df["patient_id"].iloc[0]
    g = df[df.patient_id == pid].tail(30).copy()
    latest = g.iloc[-1]
    level = latest.get("true_label", "Normal")
    plt.figure(figsize=(10, 5))
    ax = plt.gca(); ax.axis("off")
    ax.text(0.5, 0.96, "SmartHealth-IoT Remote Patient Monitoring Dashboard", ha="center", fontsize=13, fontweight="bold")
    card_text = f"Patient: {pid}\nHeart Rate: {latest.heart_rate:.1f} bpm\nSpO₂: {latest.spo2:.1f}%\nTemperature: {latest.temperature:.2f} °C\nActivity: {latest.activity:.1f}\nRisk Status: {level}"
    ax.text(0.05, 0.75, card_text, fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    inset = plt.axes([0.42, 0.18, 0.5, 0.55])
    inset.plot(range(len(g)), g["heart_rate"], label="HR")
    inset.plot(range(len(g)), g["spo2"], label="SpO₂")
    inset.plot(range(len(g)), g["temperature"] * 2.5, label="Temp×2.5")
    inset.set_title("Recent 30-Sample Physiological Trends")
    inset.set_xlabel("Recent Samples")
    inset.legend(fontsize=8)
    _save(project_root() / out)
