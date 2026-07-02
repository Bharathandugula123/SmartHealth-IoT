from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from .utils import BOUNDS, classify_vital_state, set_seed

EVENT_TYPES = ["tachycardia", "bradycardia", "hypoxia", "fever", "hypothermia", "combined"]

def _baseline(rng: np.random.Generator) -> Dict[str, float]:
    return {
        "heart_rate": rng.uniform(68, 88),
        "spo2": rng.uniform(96, 99),
        "temperature": rng.uniform(36.4, 37.1),
        "activity": rng.uniform(15, 65),
    }

def _clip(vital: str, value: float) -> float:
    lo, hi = BOUNDS[vital]["clip"]
    return float(np.clip(value, lo, hi))

def _choose_events(rng: np.random.Generator, packets: int, p_abnormal: float, min_events: int, max_events: int, min_dur: int, max_dur: int) -> List[Tuple[int, int, str, str]]:
    events = []
    if rng.random() > p_abnormal:
        return events
    n_events = int(rng.integers(min_events, max_events + 1))
    for _ in range(n_events):
        duration = int(rng.integers(min_dur, max_dur + 1))
        start = int(rng.integers(10, max(11, packets - duration - 5)))
        etype = rng.choice(EVENT_TYPES)
        severity = rng.choice(["Warning", "Critical"], p=[0.58, 0.42])
        events.append((start, min(packets - 1, start + duration), etype, severity))
    return events

def _event_target(etype: str, severity: str, base: Dict[str, float], rng: np.random.Generator) -> Dict[str, float]:
    target = base.copy()
    if etype in ("tachycardia", "combined"):
        target["heart_rate"] = rng.uniform(105, 118) if severity == "Warning" else rng.uniform(124, 150)
    if etype == "bradycardia":
        target["heart_rate"] = rng.uniform(52, 58) if severity == "Warning" else rng.uniform(38, 48)
    if etype in ("hypoxia", "combined"):
        target["spo2"] = rng.uniform(91, 94) if severity == "Warning" else rng.uniform(82, 89)
    if etype in ("fever", "combined"):
        target["temperature"] = rng.uniform(37.7, 38.3) if severity == "Warning" else rng.uniform(38.6, 40.0)
    if etype == "hypothermia":
        target["temperature"] = rng.uniform(35.6, 36.0) if severity == "Warning" else rng.uniform(34.7, 35.4)
    target["activity"] = float(np.clip(target["activity"] + rng.normal(10, 18), 0, 100))
    return target

def generate_virtual_streams(
    n_patients: int = 100,
    packets_per_patient: int = 300,
    sampling_interval_seconds: int = 1,
    abnormal_event_probability: float = 0.45,
    min_events_per_abnormal_patient: int = 1,
    max_events_per_abnormal_patient: int = 3,
    event_min_duration: int = 25,
    event_max_duration: int = 80,
    seed: int = 42,
    start_time: str | None = None,
) -> pd.DataFrame:
    """Generate software-emulated smartwatch streams for multiple virtual patients."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    base_time = datetime.fromisoformat(start_time) if start_time else datetime(2026, 1, 1, 8, 0, 0)
    rows = []
    for pid in range(1, n_patients + 1):
        patient_id = f"P-{pid:03d}"; device_id = f"D-{pid:03d}"
        vals = _baseline(rng)
        baseline = vals.copy()
        events = _choose_events(rng, packets_per_patient, abnormal_event_probability,
                                min_events_per_abnormal_patient, max_events_per_abnormal_patient,
                                event_min_duration, event_max_duration)
        event_targets = {(s, e, t, sev): _event_target(t, sev, baseline, rng) for s, e, t, sev in events}
        for i in range(packets_per_patient):
            current_target = baseline.copy()
            active_event = "none"
            active_severity = "Normal"
            for ev, target in event_targets.items():
                s, e, t, sev = ev
                if s <= i <= e:
                    current_target = target
                    active_event = t
                    active_severity = sev
                    break
            # Smooth autoregressive update toward baseline/event target plus bounded noise.
            for vital, noise in [("heart_rate", 1.8), ("spo2", 0.45), ("temperature", 0.08), ("activity", 5.0)]:
                drift = 0.13 * (current_target[vital] - vals[vital])
                vals[vital] = _clip(vital, vals[vital] + drift + rng.normal(0, noise))
            timestamp = base_time + timedelta(seconds=i * sampling_interval_seconds)
            row = {
                "patient_id": patient_id,
                "device_id": device_id,
                "timestamp": timestamp.isoformat(sep=" "),
                "heart_rate": round(vals["heart_rate"], 2),
                "spo2": round(vals["spo2"], 2),
                "temperature": round(vals["temperature"], 2),
                "activity": round(vals["activity"], 2),
                "event_type": active_event,
                "event_target_severity": active_severity,
            }
            row["true_label"] = classify_vital_state(row)
            rows.append(row)
    return pd.DataFrame(rows)
