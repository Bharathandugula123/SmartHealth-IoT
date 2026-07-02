from __future__ import annotations
import pandas as pd
from .utils import SIGNAL_COLUMNS

def smooth_patient_stream(df: pd.DataFrame, smoothing_size: int = 5) -> pd.DataFrame:
    out = df.copy()
    for col in SIGNAL_COLUMNS:
        out[col] = out.groupby("patient_id")[col].transform(lambda s: s.rolling(smoothing_size, min_periods=1).mean())
    return out
