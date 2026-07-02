from __future__ import annotations
import os, json, random
from pathlib import Path
from typing import Any, Dict
import numpy as np
import yaml

LABEL_TO_ID = {"Normal": 0, "Warning": 1, "Critical": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

SIGNAL_COLUMNS = ["heart_rate", "spo2", "temperature", "activity"]

BOUNDS = {
    "heart_rate": {"normal": (60, 100), "warning_low": (50, 59), "warning_high": (101, 120), "critical_low": (-1e9, 49.999), "critical_high": (120.001, 1e9), "clip": (35, 165)},
    "spo2": {"normal": (95, 100), "warning_low": (90, 94.999), "critical_low": (-1e9, 89.999), "clip": (75, 100)},
    "temperature": {"normal": (36.1, 37.5), "warning_high": (37.6, 38.499), "critical_low": (-1e9, 35.499), "critical_high": (38.5, 1e9), "clip": (34.5, 40.5)},
    "activity": {"normal": (0, 100), "clip": (0, 100)},
}

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def ensure_dirs() -> None:
    for d in ["data/generated", "data/features", "data/results", "figures", "models"]:
        (project_root() / d).mkdir(parents=True, exist_ok=True)

def load_config(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else project_root() / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def save_json(obj: Any, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)

def classify_vital_state(row: Dict[str, float]) -> str:
    hr = float(row["heart_rate"]); spo2 = float(row["spo2"]); temp = float(row["temperature"])
    critical = (hr > 120 or hr < 50 or spo2 < 90 or temp >= 38.5 or temp < 35.5)
    warning = ((101 <= hr <= 120) or (50 <= hr < 60) or (90 <= spo2 < 95) or (37.6 <= temp < 38.5))
    if critical:
        return "Critical"
    if warning:
        return "Warning"
    return "Normal"

def severity_max(labels):
    vals = [LABEL_TO_ID[str(x)] if isinstance(x, str) else int(x) for x in labels]
    return ID_TO_LABEL[max(vals)]
