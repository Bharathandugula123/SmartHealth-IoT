from __future__ import annotations
import numpy as np
import pandas as pd
from .utils import SIGNAL_COLUMNS, severity_max
from .preprocessing import smooth_patient_stream

def _slope(values: np.ndarray) -> float:
    values = values.astype(float)
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = (n - 1) / 2.0
    y_mean = float(values.mean())
    denom = float(((x - x_mean) ** 2).sum())
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (values - y_mean)).sum() / denom)

def extract_sliding_window_features(df: pd.DataFrame, window_size: int = 30, smoothing_size: int = 5, step_size: int = 1) -> pd.DataFrame:
    """Build leakage-safe feature rows. Split later by patient_id, not by window rows."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    smoothed = smooth_patient_stream(df, smoothing_size)
    rows = []
    for pid, g in smoothed.groupby("patient_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        for end in range(window_size - 1, len(g), step_size):
            w = g.iloc[end - window_size + 1:end + 1]
            feat = {
                "patient_id": pid,
                "device_id": str(w["device_id"].iloc[-1]),
                "window_start": w["timestamp"].iloc[0],
                "window_end": w["timestamp"].iloc[-1],
            }
            for col in SIGNAL_COLUMNS:
                vals = w[col].astype(float).values
                feat[f"{col}_mean"] = float(np.mean(vals))
                feat[f"{col}_std"] = float(np.std(vals, ddof=0))
                feat[f"{col}_min"] = float(np.min(vals))
                feat[f"{col}_max"] = float(np.max(vals))
                feat[f"{col}_slope"] = _slope(vals)
                feat[f"{col}_rate_change"] = float(vals[-1] - vals[0])
            feat["label"] = severity_max(w["true_label"].tolist())
            feat["current_label"] = str(w["true_label"].iloc[-1])
            feat["timestamp"] = w["timestamp"].iloc[-1]
            rows.append(feat)
    return pd.DataFrame(rows)

def get_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    excluded = {"patient_id", "device_id", "window_start", "window_end", "timestamp", "label", "current_label"}
    return [c for c in feature_df.columns if c not in excluded]
