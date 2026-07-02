from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.model_selection import train_test_split
from .utils import LABEL_TO_ID

def patient_level_split(feature_df: pd.DataFrame, train_ratio=0.70, validation_ratio=0.15, test_ratio=0.15, seed=42):
    patients = np.array(sorted(feature_df["patient_id"].unique()))
    train_p, temp_p = train_test_split(patients, train_size=train_ratio, random_state=seed, shuffle=True)
    val_fraction_of_temp = validation_ratio / (validation_ratio + test_ratio)
    val_p, test_p = train_test_split(temp_p, train_size=val_fraction_of_temp, random_state=seed, shuffle=True)
    train_df = feature_df[feature_df.patient_id.isin(train_p)].reset_index(drop=True)
    val_df = feature_df[feature_df.patient_id.isin(val_p)].reset_index(drop=True)
    test_df = feature_df[feature_df.patient_id.isin(test_p)].reset_index(drop=True)
    return train_df, val_df, test_df

def labels_to_ids(series):
    return np.array([LABEL_TO_ID[str(x)] for x in series])

def summarize_repeated_results(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in {"seed", "model"}]
    out = df.groupby("model")[metric_cols].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()

def pairwise_significance(run_metric_df: pd.DataFrame, proposed="VitalBoostNet", metric="macro_f1") -> pd.DataFrame:
    rows = []
    models = [m for m in run_metric_df["model"].unique() if m != proposed]
    pvals = []
    for m in models:
        a = run_metric_df[run_metric_df.model == proposed].sort_values("seed")[metric].values
        b = run_metric_df[run_metric_df.model == m].sort_values("seed")[metric].values
        n = min(len(a), len(b)); a = a[:n]; b = b[:n]
        if n < 2:
            t_p = np.nan; w_p = np.nan
        else:
            try: t_p = float(ttest_rel(a, b).pvalue)
            except Exception: t_p = np.nan
            try: w_p = float(wilcoxon(a, b).pvalue)
            except Exception: w_p = np.nan
        rows.append({"proposed": proposed, "baseline": m, "metric": metric, "mean_difference": float(np.mean(a - b)) if n else np.nan, "paired_ttest_p": t_p, "wilcoxon_p": w_p})
    return pd.DataFrame(rows)
