from __future__ import annotations
import time
import numpy as np
from typing import Dict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.dummy import DummyClassifier
from .vitalboostnet import evaluate_classifier

XGBClassifier = None
HAS_XGB = False

def build_baseline_models(seed: int = 42) -> Dict:
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        "Linear SVM-SGD": make_pipeline(StandardScaler(), SGDClassifier(loss="hinge", class_weight="balanced", random_state=seed, max_iter=1000, tol=1e-3)),
        "Random Forest": RandomForestClassifier(n_estimators=60, class_weight="balanced", random_state=seed, n_jobs=1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=60, class_weight="balanced", random_state=seed, n_jobs=1),
        "Majority Dummy": DummyClassifier(strategy="most_frequent"),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.9,
                                           objective="multi:softprob", eval_metric="mlogloss", random_state=seed, n_jobs=1)
    return models

def fit_and_evaluate_baselines(X_train, y_train, X_test, y_test, seed: int = 42):
    rows = []
    trained = {}
    for name, model in build_baseline_models(seed).items():
        model.fit(X_train, y_train)
        t0 = time.perf_counter()
        pred = model.predict(X_test)
        latency = (time.perf_counter() - t0) * 1000 / max(1, len(X_test))
        proba = None
        if hasattr(model, "predict_proba"):
            try: proba = model.predict_proba(X_test)
            except Exception: proba = None
        rows.append(evaluate_classifier(y_test, pred, proba, name, latency))
        trained[name] = model
    return rows, trained
