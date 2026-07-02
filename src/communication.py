from __future__ import annotations
import time
from typing import Dict, Tuple
from .device_emulator import REQUIRED_PACKET_FIELDS, REQUIRED_VITALS
from .utils import BOUNDS

class IngestionLayer:
    """Function-based HTTP/MQTT-style emulation with packet validation."""
    def __init__(self):
        self.valid_packets = 0
        self.dropped_packets = 0
        self.latencies_ms = []

    def validate_packet(self, packet: Dict) -> Tuple[bool, str]:
        if not REQUIRED_PACKET_FIELDS.issubset(packet.keys()):
            return False, "missing_packet_field"
        vitals = packet.get("vitals", {})
        if not REQUIRED_VITALS.issubset(vitals.keys()):
            return False, "missing_vital_field"
        for vital in REQUIRED_VITALS:
            try:
                val = float(vitals[vital])
            except Exception:
                return False, f"non_numeric_{vital}"
            lo, hi = BOUNDS[vital]["clip"]
            if not (lo <= val <= hi):
                return False, f"out_of_operational_range_{vital}"
        return True, "valid"

    def ingest(self, packet: Dict, storage=None) -> Dict:
        t0 = time.perf_counter()
        ok, reason = self.validate_packet(packet)
        if ok:
            self.valid_packets += 1
            if storage is not None:
                storage.insert_vital_packet(packet)
        else:
            self.dropped_packets += 1
        latency_ms = (time.perf_counter() - t0) * 1000
        self.latencies_ms.append(latency_ms)
        return {"valid": ok, "reason": reason, "latency_ms": latency_ms}

    def summary(self) -> Dict:
        total = self.valid_packets + self.dropped_packets
        avg_latency = sum(self.latencies_ms) / max(1, len(self.latencies_ms))
        return {
            "total_packets": total,
            "valid_packets": self.valid_packets,
            "dropped_packets": self.dropped_packets,
            "avg_ingestion_latency_ms": avg_latency,
            "drop_rate": self.dropped_packets / max(1, total),
        }
