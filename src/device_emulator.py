from __future__ import annotations
from typing import Dict, Iterator
import pandas as pd

REQUIRED_PACKET_FIELDS = {"device_id", "patient_id", "timestamp", "vitals"}
REQUIRED_VITALS = {"heart_rate", "spo2", "temperature", "activity"}

class VirtualWearableDevice:
    def __init__(self, patient_id: str, device_id: str, patient_df: pd.DataFrame):
        self.patient_id = patient_id
        self.device_id = device_id
        self.patient_df = patient_df.sort_values("timestamp").reset_index(drop=True)

    def packets(self) -> Iterator[Dict]:
        for _, r in self.patient_df.iterrows():
            yield {
                "device_id": self.device_id,
                "patient_id": self.patient_id,
                "timestamp": str(r["timestamp"]),
                "vitals": {
                    "heart_rate": float(r["heart_rate"]),
                    "spo2": float(r["spo2"]),
                    "temperature": float(r["temperature"]),
                    "activity": float(r["activity"]),
                },
                "true_label": str(r.get("true_label", "Unknown")),
                "event_type": str(r.get("event_type", "none")),
            }

def build_devices(stream_df: pd.DataFrame) -> list[VirtualWearableDevice]:
    devices = []
    for (pid, did), g in stream_df.groupby(["patient_id", "device_id"]):
        devices.append(VirtualWearableDevice(pid, did, g))
    return devices
