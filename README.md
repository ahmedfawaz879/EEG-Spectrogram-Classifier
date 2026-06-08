#  EEG Seizure Detection using Spectrograms and Deep Learning

[![CI](https://github.com/ahmedfawaz879/EEG-Spectrogram-Classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmedfawaz879/EEG-Spectrogram-Classifier/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **End-to-end deep learning pipeline** for epileptic seizure detection from EEG recordings using mel-spectrogram features, convolutional autoencoder pretraining, and CNN classification.

---

##  Quick Start

```bash
# 1. Clone & install
git clone https://github.com/ahmedfawaz879/EEG-Spectrogram-Classifier.git
cd EEG-Spectrogram-Classifier
pip install -e ".[dev]"

# 2. Train (place CHB-MIT EDF files in data/chb-mit/)
eeg-train --config configs/default.yaml

# 3. Predict
eeg-predict recording.edf --checkpoint runs/clf.pth

# 4. Launch demo
pip install -e ".[demo]"
python -m eeg_classifier.demo.app
```

---

##  Architecture

```
EDF File → Windowing → Mel Spectrogram → Autoencoder Pretraining → CNN Classifier → Seizure Prediction
```

```
┌─────────┐    ┌──────────┐    ┌───────────────┐    ┌───────────┐    ┌────────────┐
│ EDF     │───▶│ Sliding  │───▶│ Mel           │───▶│ Conv      │───▶│ Classifier │───▶ Seizure / Non-Seizure
│ Loader  │    │ Window   │    │ Spectrogram   │    │ Autoenc.  │    │ Head       │
└─────────┘    └──────────┘    └───────────────┘    └───────────┘    └────────────┘
     ▲              │                │                    │                │
     │         10s windows      (C, 64, T)          128-d latent      2-class logits
  MNE-Python   5s stride        0.5–40 Hz            embedding
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| **Subject-aware splitting** | `GroupShuffleSplit` by patient ID prevents data leakage — a common pitfall in EEG studies |
| **Autoencoder pretraining** | Self-supervised representation learning improves generalisation on small labelled datasets |
| **Mel-spectrogram features** | Captures time-frequency patterns relevant to seizure morphology (0.5–40 Hz band) |
| **YAML configuration** | All hyperparameters centralised; experiments are reproducible via config files |
| **MLflow tracking** | Every training run logs metrics, artifacts, and configs automatically |

---

##  Project Structure

```
eeg-spectrogram-classifier/
│
├── src/eeg_classifier/
│   ├── data/
│   │   ├── edf_loader.py        # EDF discovery, subject-ID extraction, MNE loading
│   │   ├── windowing.py         # Sliding-window segmentation
│   │   ├── labeling.py          # Seizure annotation parsing & label assignment
│   │   └── dataset.py           # PyTorch Dataset wrapper
│   │
│   ├── features/
│   │   └── spectrogram.py       # Multi-channel mel-spectrogram extraction
│   │
│   ├── models/
│   │   ├── autoencoder.py       # Convolutional autoencoder (encoder + decoder)
│   │   └── classifier.py        # Seizure classifier (encoder + FC head)
│   │
│   ├── training/
│   │   ├── pipeline.py          # Data pipeline with subject-aware splitting
│   │   ├── train_ae.py          # Autoencoder pretraining loop
│   │   └── train.py             # Full training orchestrator (AE → CLF → Eval)
│   │
│   ├── evaluation/
│   │   └── metrics.py           # Accuracy, F1, ROC-AUC, confusion matrix plots
│   │
│   ├── inference/
│   │   └── predict.py           # Single-file inference → JSON output
│   │
│   ├── demo/
│   │   └── app.py               # Gradio interactive demo
│   │
│   └── config.py                # YAML config loader with override support
│
├── configs/
│   └── default.yaml             # Default experiment configuration
│
├── tests/                       # Unit tests (pytest)
│   ├── test_windowing.py
│   ├── test_spectrogram.py
│   ├── test_dataset.py
│   ├── test_model_shapes.py
│   ├── test_labeling.py
│   ├── test_edf_loader.py
│   └── test_config.py
│
├── notebooks/                   # Original research notebook
├── .github/workflows/ci.yml    # GitHub Actions CI pipeline
├── Dockerfile                   # Docker container
├── pyproject.toml               # Package definition & dependencies
└── README.md
```

---

##  Methodology

### Data Leakage Prevention

EEG recordings contain subject-specific patterns (electrode impedance, brain anatomy). Naïve random splitting can leak patient identity into the test set, inflating accuracy. This pipeline uses **`GroupShuffleSplit`** by patient ID:

```python
# Subject-aware split — no patient appears in both train and test
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(indices, groups=patient_ids))
```

### Signal Processing

| Parameter | Value | Rationale |
|---|---|---|
| Sample rate | 256 Hz | Standard for clinical EEG |
| Window duration | 10 s | Captures seizure onset patterns |
| Window stride | 5 s | 50% overlap for temporal context |
| Mel bins | 64 | Sufficient frequency resolution |
| Frequency range | 0.5–40 Hz | Covers delta through gamma bands |

### Model Architecture

**Autoencoder** (unsupervised pretraining):
- 3-layer conv encoder → 128-d bottleneck → 3-layer transposed-conv decoder
- BatchNorm + ReLU activation
- MSE reconstruction loss

**Classifier** (supervised fine-tuning):
- Pretrained encoder (frozen conv layers, trainable projection)
- 2-layer FC head with dropout (0.4) → 2-class softmax

---

##  Experiment Tracking

All training runs are tracked with **MLflow**:

```bash
# View experiment dashboard
mlflow ui --backend-store-uri mlruns
```

Tracked metrics:
- `ae_train_loss` — autoencoder reconstruction loss per epoch
- `clf_train_loss` / `clf_val_loss` — classifier training & validation loss
- `test_accuracy`, `test_f1`, `test_precision`, `test_recall`, `test_roc_auc`

Tracked artifacts:
- `config.yaml` — full experiment configuration
- `confusion_matrix.png` — test set confusion matrix
- Model checkpoints (`ae.pth`, `clf.pth`)

---

##  Docker

```bash
# Build
docker build -t eeg-classifier .

# Train (mount data directory)
docker run -v /path/to/edf/data:/app/data -v ./runs:/app/runs eeg-classifier

# Predict
docker run -v ./runs:/app/runs eeg-classifier \
    python -m eeg_classifier.inference.predict /app/data/sample.edf
```

---

##  Interactive Demo

```bash
pip install -e ".[demo]"
python -m eeg_classifier.demo.app
```

Upload an EDF file to see:
- 📈 Raw EEG signal visualisation
- 🎨 Mel spectrogram heatmap
- 🔴🟢 Per-window seizure predictions with confidence scores

---

##  Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=eeg_classifier --cov-report=html

# Lint
ruff check src/ tests/
```

---

##  Configuration

All hyperparameters are defined in `configs/default.yaml`. Override with a custom YAML:

```yaml
# configs/experiment_01.yaml
classifier:
  epochs: 50
  learning_rate: 5.0e-5
  dropout: 0.5

autoencoder:
  epochs: 20
```

```bash
eeg-train --config configs/experiment_01.yaml
```

---

##  Supported Datasets

| Dataset | Format | Auto-detected |
|---|---|---|
| [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/) | EDF + seizure summary | ✅ |
| [TUH Seizure Corpus](https://isip.piconepress.com/projects/tuh_eeg/) | EDF + annotations | ✅ (with CSV) |
| Custom | EDF | ✅ (with CSV annotations) |

---

##  Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA (optional, for GPU acceleration)

See `pyproject.toml` for the full dependency list.

---

##  Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/improvement`)
3. Run tests (`pytest tests/ -v`)
4. Submit a pull request

---

##  License

This project is released under the [MIT License](LICENSE).

---

##  References

- Shoeb, A. H. (2009). *Application of Machine Learning to Epileptic Seizure Detection*. MIT PhD Thesis.
- CHB-MIT Scalp EEG Database: [PhysioNet](https://physionet.org/content/chbmit/1.0.0/)
- Shah, V. et al. (2018). *The Temple University Hospital Seizure Detection Corpus*.
