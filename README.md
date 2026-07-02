https://doi.org/10.5281/zenodo.21134695
# SmartHealth-IoT

**A Simulation-Based IoT Remote Patient Monitoring Framework Using Virtual Wearable Physiological Data Streams**


## Overview

**SmartHealth-IoT** is a Python-based simulation framework for remote patient monitoring using virtual smartwatch physiological data streams.

The framework generates synthetic wearable sensor data, emulates IoT communication, extracts temporal features, trains a lightweight machine learning model (**VitalBoostNet**), and produces intelligent health alerts by combining machine learning predictions with physiological rule-based reasoning.
---

# Features

- Virtual wearable physiological signal generation
- Multi-patient simulation
- Multi-device simulation
- IoT communication emulation
- Sliding-window feature extraction
- Statistical & temporal feature engineering
- VitalBoostNet (LightGBM)
- Baseline model comparison
- Hybrid alert generation
- SQLite backend
- Scalability evaluation
- Publication-ready visualizations
- Optional Streamlit dashboard

---

# Project Structure

```text
SmartHealth-IoT/
│
├── main.py
├── app.py
├── config.yaml
├── requirements.txt
├── README.md
│
├── src/
│   ├── data_generator.py
│   ├── device_emulator.py
│   ├── communication.py
│   ├── database.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── labeling.py
│   ├── vitalboostnet.py
│   ├── baselines.py
│   ├── alert_engine.py
│   ├── evaluation.py
│   ├── scalability_test.py
│   ├── visualization.py
│   └── utils.py
│
├── data/
│   ├── generated/
│   ├── features/
│   └── results/
│
├── figures/
│
└── models/
```

---

# Methodology

The framework follows six major stages.

```text
Virtual Physiological Data Generation
            │
            ▼
Wearable Device Simulation
            │
            ▼
IoT Communication Emulation
            │
            ▼
Feature Engineering
            │
            ▼
VitalBoostNet Prediction
            │
            ▼
Hybrid Alert Generation
```

---

# Simulated Physiological Signals

| Signal | Description |
|---------|-------------|
| Heart Rate | Beats Per Minute |
| SpO₂ | Blood Oxygen Saturation |
| Temperature | Body Temperature (°C) |
| Activity | Simulated Activity Index |

---

# Risk Classes

| Label | Class |
|------|---------|
| 0 | Normal |
| 1 | Warning |
| 2 | Critical |

---

# Installation

Clone the repository

```bash
git clone https://github.com/your-username/SmartHealth-IoT.git

cd SmartHealth-IoT
```

Create a virtual environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Project

```bash
python main.py
```

This automatically performs:

- Data simulation
- Feature extraction
- Model training
- Baseline comparison
- Evaluation
- Alert generation
- Figure generation
- Database creation

---

# Launch Dashboard

```bash
streamlit run app.py
```

Dashboard includes

- Patient monitoring
- Live physiological trends
- Risk prediction
- Alert history
- Performance summary
- Scalability visualization

---

# Configuration

Modify

```text
config.yaml
```

Example

```yaml
simulation:
  num_patients: 100
  packets_per_patient: 300
  random_seed: 42

features:
  window_size: 30

vitalboostnet:
  learning_rate: 0.05
  n_estimators: 300
```

---

# VitalBoostNet

VitalBoostNet is a lightweight LightGBM multiclass classifier.

Extracted features include

- Mean
- Standard Deviation
- Minimum
- Maximum
- Trend Slope
- Rate of Change

---

# Baseline Models

- Logistic Regression
- Support Vector Machine
- Random Forest
- XGBoost (Optional)
- Rule-Based Classifier

---

# Evaluation Metrics

### Classification

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

### Alert Evaluation

- Event Sensitivity
- False Alert Rate
- Missed Event Rate
- Mean Alert Delay

### System Performance

- CPU Usage
- Memory Usage
- Throughput
- Inference Latency
- Scalability

---

# Output Files

```text
data/generated/virtual_patient_streams.csv

data/features/smarthealth_features.csv

data/results/classification_results.csv

data/results/baseline_results.csv

data/results/alert_results.csv

data/results/scalability_results.csv

models/vitalboostnet.pkl

models/scaler.pkl

figures/confusion_matrix.png

figures/model_comparison.png

figures/roc_auc_curve.png

figures/alert_performance.png
```

---

# Reproducibility

The framework supports reproducible experimentation through

- Fixed random seeds
- Patient-level data splitting
- Configurable simulations
- Saved datasets
- Saved models
- Automatic figure generation

Recommended Seeds

```
42
52
62
72
82
```

---

# Example Workflow

```bash
pip install -r requirements.txt

python main.py

streamlit run app.py
```

---

# Research Notes

- No real patient data are used.
- No smartwatch hardware is required.
- Communication is software-emulated.
- Thresholds are simulation-based.
- Results are intended for research only.

---

# Suggested Citation

```bibtex
@article{smarthealthiot2026,
  title={SmartHealth-IoT: A Simulation-Based IoT Remote Patient Monitoring Framework Using Virtual Wearable Physiological Data Streams},
  author={bharathandugula},
  journal={Under Review},
  year={2026}
}
```

---

# License

This project is released for academic and research purposes.

You may choose one of the following licenses:

- MIT
- Apache 2.0
- GPL-3.0

---


