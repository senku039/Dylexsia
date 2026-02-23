# 5_Production_ML

Production-ready ML module for dyslexia screening from eye-tracking data.

## Files

- `data.py` — CSV loading, schema validation (`LX`, `LY`, `RX`, `RY`), label encoding (`0=Control`, `1=Dyslexic`), and group extraction.
- `features.py` — robust fixed-length feature extraction from variable-length sequences:
  - temporal features (position, velocity, acceleration statistics)
  - FFT bandpower + spectral entropy
  - STFT summary features
  - NaN/Inf-safe output
- `model.py` — grouped nested CV evaluation (`StratifiedGroupKFold`) with:
  - leakage-safe sklearn `Pipeline`
  - inner-loop `GridSearchCV` hyperparameter tuning
  - probability calibration (`platt` / `isotonic` / `none`)
  - threshold optimization (Youden's J)
  - unbiased outer-fold metrics + 95% confidence intervals
- `train.py` — CLI runner for real data or synthetic smoke tests.

## Data format

Expected directory structure:

```text
<DataDir>/
  Control/
    *.csv
  Dyslexic/
    *.csv
```

Each CSV must contain:

- `LX`, `LY`, `RX`, `RY`

## CLI usage

From repository root:

```bash
python '5_Production_ML/train.py' --data-dir Data --model logreg --splits 5 --repeats 2 --random-state 42 --calibration platt
```

Smoke test (synthetic grouped data):

```bash
python '5_Production_ML/train.py' --smoke-test --model rf --splits 5 --repeats 2 --random-state 42 --calibration isotonic
```

## Reported metrics

- Accuracy
- Balanced Accuracy
- Sensitivity (Recall for Dyslexic)
- Specificity
- F1
- MCC
- ROC-AUC
- PR-AUC

Each metric is summarized with mean, std, and 95% CI from outer folds.

## Python version

- Designed for Python 3.11
