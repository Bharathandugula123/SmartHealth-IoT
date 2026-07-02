from __future__ import annotations
import pandas as pd
from .utils import classify_vital_state, LABEL_TO_ID, ID_TO_LABEL

def assign_point_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["true_label"] = out.apply(lambda r: classify_vital_state(r.to_dict()), axis=1)
    return out

def encode_labels(labels):
    return [LABEL_TO_ID[str(x)] for x in labels]

def decode_labels(ids):
    return [ID_TO_LABEL[int(x)] for x in ids]
