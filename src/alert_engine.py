from __future__ import annotations
import time
import numpy as np
import pandas as pd
from .utils import classify_vital_state, LABEL_TO_ID, ID_TO_LABEL

class HybridAlertEngine:
    def __init__(self, alpha: float = 0.50, strategy: str = "or"):
        self.alpha = alpha
        self.strategy = strategy

    @staticmethod
    def rule_level(row) -> str:
        return classify_vital_state(row)

    def fuse(self, rule_level: str, model_label: str) -> tuple[int, str, str]:
        rule_score = LABEL_TO_ID[rule_level]
        model_score = LABEL_TO_ID[model_label]
        if self.strategy == "weighted":
            score = self.alpha * rule_score + (1 - self.alpha) * model_score
            final_score = 2 if score >= 1.5 else (1 if score >= 0.5 else 0)
        else:
            final_score = max(rule_score, model_score)
        final_level = ID_TO_LABEL[final_score]
        trigger = []
        if rule_score > 0: trigger.append(f"rule:{rule_level}")
        if model_score > 0: trigger.append(f"model:{model_label}")
        return int(final_score > 0), final_level, ";".join(trigger) if trigger else "none"

    def generate_alerts(self, feature_df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
        rows = []
        for i, (_, r) in enumerate(feature_df.iterrows()):
            t0 = time.perf_counter()
            raw_row = {
                "heart_rate": r.get("heart_rate_mean", r.get("heart_rate", 0)),
                "spo2": r.get("spo2_mean", r.get("spo2", 0)),
                "temperature": r.get("temperature_mean", r.get("temperature", 0)),
                "activity": r.get("activity_mean", r.get("activity", 0)),
            }
            rule_label = self.rule_level(raw_row)
            model_label = ID_TO_LABEL[int(predictions[i])]
            final_alert, final_level, trigger = self.fuse(rule_label, model_label)
            rows.append({
                "patient_id": r["patient_id"],
                "timestamp": str(r["timestamp"]),
                "true_label": r["label"],
                "rule_alert": int(LABEL_TO_ID[rule_label] > 0),
                "model_alert": int(LABEL_TO_ID[model_label] > 0),
                "final_alert": final_alert,
                "rule_level": rule_label,
                "model_level": model_label,
                "alert_level": final_level,
                "triggering_condition": trigger,
                "alert_latency_ms": (time.perf_counter() - t0) * 1000,
            })
        return pd.DataFrame(rows)

def evaluate_alerts(alert_df: pd.DataFrame) -> dict:
    truth_alert = alert_df["true_label"].map(lambda x: 0 if str(x) == "Normal" else 1).astype(int).values
    pred_alert = alert_df["final_alert"].astype(int).values
    tp = int(((truth_alert == 1) & (pred_alert == 1)).sum())
    tn = int(((truth_alert == 0) & (pred_alert == 0)).sum())
    fp = int(((truth_alert == 0) & (pred_alert == 1)).sum())
    fn = int(((truth_alert == 1) & (pred_alert == 0)).sum())
    event_sensitivity = tp / max(1, tp + fn)
    false_alert_rate = fp / max(1, fp + tn)
    missed_event_rate = fn / max(1, tp + fn)
    return {
        "event_sensitivity": event_sensitivity,
        "false_alert_rate": false_alert_rate,
        "missed_event_rate": missed_event_rate,
        "mean_alert_latency_ms": float(alert_df["alert_latency_ms"].mean()),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
