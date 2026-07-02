from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Dict, Iterable
import pandas as pd

class SmartHealthDB:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.create_tables()

    def create_tables(self) -> None:
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS vitals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT, device_id TEXT, timestamp TEXT,
            heart_rate REAL, spo2 REAL, temperature REAL, activity REAL,
            true_label TEXT, event_type TEXT
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT, timestamp TEXT,
            predicted_label TEXT, probability REAL, inference_latency_ms REAL
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT, timestamp TEXT,
            rule_alert INTEGER, model_alert INTEGER, final_alert INTEGER,
            alert_level TEXT, triggering_condition TEXT, alert_latency_ms REAL
        )""")
        self.conn.commit()

    def insert_vital_packet(self, packet: Dict) -> None:
        v = packet["vitals"]
        self.conn.execute("""INSERT INTO vitals(patient_id,device_id,timestamp,heart_rate,spo2,temperature,activity,true_label,event_type)
                         VALUES (?,?,?,?,?,?,?,?,?)""",
                         (packet["patient_id"], packet["device_id"], packet["timestamp"], v["heart_rate"], v["spo2"], v["temperature"], v["activity"], packet.get("true_label"), packet.get("event_type")))
        self.conn.commit()

    def insert_alerts_df(self, df: pd.DataFrame) -> None:
        keep = ["patient_id", "timestamp", "rule_alert", "model_alert", "final_alert", "alert_level", "triggering_condition", "alert_latency_ms"]
        df[keep].to_sql("alerts", self.conn, if_exists="append", index=False)

    def read_table(self, table: str) -> pd.DataFrame:
        return pd.read_sql_query(f"SELECT * FROM {table}", self.conn)

    def close(self) -> None:
        self.conn.close()
