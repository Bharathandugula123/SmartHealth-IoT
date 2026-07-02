from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from src.utils import load_config, ensure_dirs, project_root, LABEL_TO_ID, ID_TO_LABEL, save_json
from src.data_generator import generate_virtual_streams
from src.device_emulator import build_devices
from src.communication import IngestionLayer
from src.database import SmartHealthDB
from src.feature_engineering import extract_sliding_window_features, get_feature_columns
from src.evaluation import patient_level_split, labels_to_ids, summarize_repeated_results, pairwise_significance
from src.vitalboostnet import VitalBoostNet, evaluate_classifier
from src.baselines import fit_and_evaluate_baselines
from src.alert_engine import HybridAlertEngine, evaluate_alerts
from src.scalability_test import run_scalability_test
from src.visualization import plot_sample_stream, plot_model_comparison, plot_confusion, plot_multiclass_roc, plot_alert_metrics, plot_scalability, make_dashboard_snapshot

def run_pipeline(config_path=None):
    ensure_dirs()
    root = project_root()
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]
    seeds = cfg["project"]["random_seeds"]

    print("[1/10] Generating virtual wearable physiological streams...")
    stream_df = generate_virtual_streams(**{k:v for k,v in data_cfg.items() if k not in {"output_csv"}}, seed=seeds[0])
    stream_path = root / data_cfg["output_csv"]
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    stream_df.to_csv(stream_path, index=False)

    print("[2/10] Emulating virtual devices and IoT ingestion...")
    db = SmartHealthDB(root / "data/results/smarthealth_iot.sqlite")
    ing = IngestionLayer()
    for dev in build_devices(stream_df):
        for packet in dev.packets():
            ing.ingest(packet, storage=db)
    save_json(ing.summary(), root / "data/results/ingestion_summary.json")

    print("[3/10] Extracting sliding-window features...")
    feature_df = extract_sliding_window_features(stream_df, window_size=feat_cfg["window_size"], smoothing_size=feat_cfg["smoothing_size"], step_size=feat_cfg["step_size"])
    feat_path = root / feat_cfg["output_csv"]
    feature_df.to_csv(feat_path, index=False)
    feature_cols = get_feature_columns(feature_df)

    print("[4/10] Training VitalBoostNet and baseline models across repeated seeds...")
    all_metric_rows = []
    final_artifacts = {}
    for seed in seeds:
        train_df, val_df, test_df = patient_level_split(feature_df, cfg["splitting"]["train_ratio"], cfg["splitting"]["validation_ratio"], cfg["splitting"]["test_ratio"], seed)
        X_train = train_df[feature_cols].values; y_train = labels_to_ids(train_df["label"])
        X_val = val_df[feature_cols].values; y_val = labels_to_ids(val_df["label"])
        X_test = test_df[feature_cols].values; y_test = labels_to_ids(test_df["label"])

        vb = VitalBoostNet(cfg["model"]["vitalboostnet"], seed=seed)
        vb.fit(X_train, y_train, X_val, y_val)
        pred, proba, latency = vb.timed_predict(X_test)
        row = evaluate_classifier(y_test, pred, proba, "VitalBoostNet", latency); row["seed"] = seed
        all_metric_rows.append(row)

        baseline_rows, _ = fit_and_evaluate_baselines(X_train, y_train, X_test, y_test, seed=seed)
        for br in baseline_rows:
            br["seed"] = seed
            all_metric_rows.append(br)

        if seed == seeds[0]:
            vb.save(root / "models/vitalboostnet.pkl", root / "models/scaler.pkl")
            final_artifacts = {"test_df": test_df, "y_test": y_test, "pred": pred, "proba": proba, "feature_cols": feature_cols}

    metric_df = pd.DataFrame(all_metric_rows)
    metric_df.to_csv(root / "data/results/classification_results_all_runs.csv", index=False)
    summary_df = summarize_repeated_results(all_metric_rows)
    summary_df.to_csv(root / "data/results/classification_results_summary.csv", index=False)

    print("[5/10] Computing statistical significance against baselines...")
    sig_df = pairwise_significance(metric_df, proposed="VitalBoostNet", metric="macro_f1")
    sig_df.to_csv(root / "data/results/statistical_significance_macro_f1.csv", index=False)

    print("[6/10] Evaluating hybrid alert generation...")
    alert_engine = HybridAlertEngine(alpha=cfg["alert"]["fusion_alpha"], strategy=cfg["alert"]["primary_strategy"])
    alert_df = alert_engine.generate_alerts(final_artifacts["test_df"], final_artifacts["pred"])
    alert_metrics = evaluate_alerts(alert_df)
    alert_df.to_csv(root / "data/results/hybrid_alert_log.csv", index=False)
    pd.DataFrame([alert_metrics]).to_csv(root / "data/results/alert_results.csv", index=False)
    db.insert_alerts_df(alert_df)

    print("[7/10] Running scalability stress-test...")
    sc = cfg["scalability"]
    scale_df = run_scalability_test(sc["patient_loads"], sc["packets_per_patient"], seed=seeds[0])
    scale_df.to_csv(root / "data/results/scalability_results.csv", index=False)

    print("[8/10] Saving classification reports...")
    labels = [ID_TO_LABEL[i] for i in range(3)]
    report = classification_report(final_artifacts["y_test"], final_artifacts["pred"], labels=[0,1,2], target_names=labels, output_dict=True, zero_division=0)
    save_json(report, root / "data/results/vitalboostnet_classification_report.json")
    pd.DataFrame(confusion_matrix(final_artifacts["y_test"], final_artifacts["pred"], labels=[0,1,2]), index=labels, columns=labels).to_csv(root / "data/results/confusion_matrix.csv")

    print("[9/10] Generating 600-dpi figures...")
    plot_sample_stream(stream_df)
    plot_model_comparison(summary_df)
    plot_confusion(final_artifacts["y_test"], final_artifacts["pred"])
    plot_multiclass_roc(final_artifacts["y_test"], final_artifacts["proba"])
    plot_alert_metrics(alert_metrics)
    plot_scalability(scale_df)
    make_dashboard_snapshot(stream_df)

    print("[10/10] Done.")
    print(f"Generated files are under: {root}")
    db.close()
    return {
        "stream_path": str(stream_path),
        "feature_path": str(feat_path),
        "classification_summary": str(root / "data/results/classification_results_summary.csv"),
        "alert_results": str(root / "data/results/alert_results.csv"),
        "scalability_results": str(root / "data/results/scalability_results.csv"),
    }

if __name__ == "__main__":
    run_pipeline()
