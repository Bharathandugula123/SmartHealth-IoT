from __future__ import annotations
import time
import psutil
import pandas as pd
from .data_generator import generate_virtual_streams
from .device_emulator import build_devices
from .communication import IngestionLayer

def run_scalability_test(patient_loads, packets_per_patient=300, seed=42):
    rows = []
    process = psutil.Process()
    for load in patient_loads:
        df = generate_virtual_streams(n_patients=int(load), packets_per_patient=int(packets_per_patient), seed=seed, abnormal_event_probability=0.45)
        devices = build_devices(df)
        ing = IngestionLayer()
        cpu_samples = []
        mem_samples = []
        start = time.perf_counter()
        for dev in devices:
            for packet in dev.packets():
                ing.ingest(packet, storage=None)
                cpu_samples.append(psutil.cpu_percent(interval=None))
                mem_samples.append(process.memory_info().rss / (1024 ** 2))
        elapsed = time.perf_counter() - start
        total_packets = int(load) * int(packets_per_patient)
        rows.append({
            "patient_load": int(load),
            "packets": total_packets,
            "elapsed_seconds": elapsed,
            "throughput_packets_per_second": total_packets / max(elapsed, 1e-9),
            "avg_ingestion_latency_ms": ing.summary()["avg_ingestion_latency_ms"],
            "cpu_usage_percent_mean": float(pd.Series(cpu_samples).mean()) if cpu_samples else 0.0,
            "memory_usage_mb_mean": float(pd.Series(mem_samples).mean()) if mem_samples else 0.0,
        })
    return pd.DataFrame(rows)
