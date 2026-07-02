from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, Tuple
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from joblib import dump, load
from .utils import LABEL_TO_ID, ID_TO_LABEL

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except Exception:
    LGBMClassifier = None
    HAS_LIGHTGBM = False

class VitalBoostNet:
    """Optimized LightGBM-based lightweight risk classifier with sklearn fallback."""
    def __init__(self, params: Dict | None = None, seed: int = 42):
        self.params = params or {}
        self.seed = seed
        self.scaler = StandardScaler()
        self.model = self._build_model()
        self.uses_lightgbm = bool(HAS_LIGHTGBM and os.getenv("SMARTHEALTH_USE_LIGHTGBM", "0") == "1")

    def _build_model(self):
        if HAS_LIGHTGBM and os.getenv("SMARTHEALTH_USE_LIGHTGBM", "0") == "1":
            return LGBMClassifier(
                objective="multiclass",
                num_class=3,
                learning_rate=self.params.get("learning_rate", 0.05),
                num_leaves=self.params.get("num_leaves", 31),
                max_depth=self.params.get("max_depth", 7),
                n_estimators=self.params.get("n_estimators", 300),
                min_child_samples=self.params.get("min_child_samples", 20),
                colsample_bytree=self.params.get("feature_fraction", 0.90),
                subsample=self.params.get("bagging_fraction", 0.80),
                subsample_freq=self.params.get("bagging_freq", 5),
                reg_alpha=self.params.get("reg_alpha", 0.01),
                reg_lambda=self.params.get("reg_lambda", 0.10),
                class_weight="balanced",
                random_state=self.seed,
                n_jobs=-1,
                verbosity=-1,
            )
        return RandomForestClassifier(n_estimators=min(int(self.params.get("n_estimators", 120)), 80), max_depth=self.params.get("max_depth", 7), class_weight="balanced", random_state=self.seed, n_jobs=1)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.scaler.fit(X_train)
        Xtr = self.scaler.transform(X_train)
        if self.uses_lightgbm and X_val is not None and y_val is not None:
            Xv = self.scaler.transform(X_val)
            try:
                from lightgbm import early_stopping, log_evaluation
                self.model.fit(Xtr, y_train, eval_set=[(Xv, y_val)], eval_metric="multi_logloss",
                               callbacks=[early_stopping(self.params.get("early_stopping_rounds", 30), verbose=False), log_evaluation(0)])
            except Exception:
                self.model.fit(Xtr, y_train)
        else:
            sw = compute_sample_weight(class_weight="balanced", y=y_train)
            try:
                self.model.fit(Xtr, y_train, sample_weight=sw)
            except TypeError:
                self.model.fit(Xtr, y_train)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        Xs = self.scaler.transform(X)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(Xs)
        preds = self.model.predict(Xs)
        out = np.zeros((len(preds), 3)); out[np.arange(len(preds)), preds] = 1.0
        return out

    def timed_predict(self, X) -> Tuple[np.ndarray, np.ndarray, float]:
        t0 = time.perf_counter()
        proba = self.predict_proba(X)
        pred = np.argmax(proba, axis=1)
        latency_ms = (time.perf_counter() - t0) * 1000 / max(1, len(X))
        return pred, proba, latency_ms

    def save(self, model_path: str | Path, scaler_path: str | Path | None = None) -> None:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        dump({"model": self.model, "scaler": self.scaler, "uses_lightgbm": self.uses_lightgbm}, model_path)
        if scaler_path:
            dump(self.scaler, scaler_path)

    @staticmethod
    def load(model_path: str | Path):
        obj = load(model_path)
        v = VitalBoostNet()
        v.model = obj["model"]; v.scaler = obj["scaler"]; v.uses_lightgbm = obj.get("uses_lightgbm", False)
        return v

def evaluate_classifier(y_true, y_pred, proba=None, model_name="model", latency_ms=None) -> Dict:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    wprec, wrec, wf1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    auc = np.nan
    if proba is not None:
        try:
            y_bin = label_binarize(y_true, classes=[0, 1, 2])
            auc = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
        except Exception:
            pass
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1,
        "weighted_precision": wprec,
        "weighted_recall": wrec,
        "weighted_f1": wf1,
        "roc_auc_macro_ovr": auc,
        "inference_latency_ms_per_sample": latency_ms if latency_ms is not None else np.nan,
    }
