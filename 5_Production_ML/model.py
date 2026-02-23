"""Grouped nested CV model evaluation with tuning, calibration, and threshold optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class EvaluationResult:
    fold_metrics: pd.DataFrame
    summary_metrics: pd.DataFrame


@dataclass
class _Calibrator:
    method: str
    model: Any


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")


def _safe_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")


def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": _specificity(y_true, y_pred),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": _safe_roc_auc(y_true, y_prob),
        "pr_auc": _safe_pr_auc(y_true, y_prob),
    }


def _build_pipeline(model_name: str, random_state: int) -> Pipeline:
    if model_name == "logreg":
        clf = LogisticRegression(
            solver="liblinear",
            max_iter=5000,
            class_weight="balanced",
            random_state=random_state,
        )
    elif model_name == "rf":
        clf = RandomForestClassifier(
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def _param_grid(model_name: str) -> list[dict[str, Any]]:
    if model_name == "logreg":
        return [{"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__penalty": ["l1", "l2"]}]
    if model_name == "rf":
        return [
            {
                "clf__n_estimators": [200, 500],
                "clf__max_depth": [None, 6, 12],
                "clf__min_samples_leaf": [1, 2, 4],
                "clf__max_features": ["sqrt", 0.5],
            }
        ]
    raise ValueError(f"Unsupported model: {model_name}")


def _fit_calibrator(y_true: np.ndarray, y_prob: np.ndarray, method: str, random_state: int) -> _Calibrator:
    if method == "none":
        return _Calibrator(method="none", model=None)
    if method == "platt":
        lr = LogisticRegression(solver="lbfgs", random_state=random_state)
        lr.fit(y_prob.reshape(-1, 1), y_true)
        return _Calibrator(method="platt", model=lr)
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(y_prob, y_true)
        return _Calibrator(method="isotonic", model=iso)
    raise ValueError(f"Unsupported calibration method: {method}")


def _apply_calibrator(cal: _Calibrator, y_prob: np.ndarray) -> np.ndarray:
    if cal.method == "none":
        return y_prob
    if cal.method == "platt":
        return cal.model.predict_proba(y_prob.reshape(-1, 1))[:, 1]
    if cal.method == "isotonic":
        return cal.model.predict(y_prob)
    raise ValueError(f"Unsupported calibration method: {cal.method}")


def _optimal_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    thresholds = np.unique(np.clip(y_prob, 0.0, 1.0))
    if len(thresholds) == 0:
        return 0.5

    best_j = -np.inf
    best_t = 0.5
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        sens = recall_score(y_true, y_pred, zero_division=0)
        spec = _specificity(y_true, y_pred)
        j = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_t = float(t)
    return best_t


def _mean_std_ci95(x: pd.Series) -> tuple[float, float, float, float]:
    arr = x.dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    hw = 1.96 * std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return mean, std, mean - hw, mean + hw


def evaluate_grouped_nested_cv(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str = "logreg",
    splits: int = 5,
    repeats: int = 1,
    random_state: int = 42,
    calibration: str = "platt",
) -> EvaluationResult:
    """Run grouped nested CV and return unbiased outer-fold metrics with 95% CIs."""
    if splits < 2:
        raise ValueError("splits must be >= 2")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    rows: list[dict[str, float]] = []
    base = _build_pipeline(model_name, random_state=random_state)
    grid = _param_grid(model_name)

    for rep in range(repeats):
        seed = random_state + rep
        outer_cv = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed)

        for fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(x, y, groups), start=1):
            x_outer_train, y_outer_train, g_outer_train = x.iloc[tr_idx], y[tr_idx], groups[tr_idx]
            x_outer_test, y_outer_test = x.iloc[te_idx], y[te_idx]

            # calibration holdout from outer-train only
            calib_cv = StratifiedGroupKFold(n_splits=min(3, splits), shuffle=True, random_state=seed + 100)
            model_idx, calib_idx = next(calib_cv.split(x_outer_train, y_outer_train, g_outer_train))

            x_model_train = x_outer_train.iloc[model_idx]
            y_model_train = y_outer_train[model_idx]
            g_model_train = g_outer_train[model_idx]
            x_calib = x_outer_train.iloc[calib_idx]
            y_calib = y_outer_train[calib_idx]

            inner_cv = StratifiedGroupKFold(n_splits=min(3, splits), shuffle=True, random_state=seed + 200)
            search = GridSearchCV(
                estimator=clone(base),
                param_grid=grid,
                scoring="roc_auc",
                cv=inner_cv,
                n_jobs=-1,
                refit=True,
            )
            search.fit(x_model_train, y_model_train, groups=g_model_train)

            best_model = clone(search.best_estimator_)
            best_model.fit(x_model_train, y_model_train)

            prob_calib_raw = best_model.predict_proba(x_calib)[:, 1]
            calibrator = _fit_calibrator(y_calib, prob_calib_raw, method=calibration, random_state=seed + 300)
            prob_calib = _apply_calibrator(calibrator, prob_calib_raw)
            threshold = _optimal_threshold_youden(y_calib, prob_calib)

            prob_test_raw = best_model.predict_proba(x_outer_test)[:, 1]
            prob_test = _apply_calibrator(calibrator, prob_test_raw)
            pred_test = (prob_test >= threshold).astype(int)

            row: dict[str, float] = {
                "repeat": float(rep + 1),
                "fold": float(fold_idx),
                "threshold": float(threshold),
                "inner_best_roc_auc": float(search.best_score_),
            }
            row.update(_metrics(y_outer_test, pred_test, prob_test))
            rows.append(row)

    fold_metrics = pd.DataFrame(rows)

    summary_rows: list[dict[str, float]] = []
    for metric in [
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
    ]:
        mean, std, lo, hi = _mean_std_ci95(fold_metrics[metric])
        summary_rows.append({"metric": metric, "mean": mean, "std": std, "ci95_low": lo, "ci95_high": hi})

    summary = pd.DataFrame(summary_rows).set_index("metric")
    return EvaluationResult(fold_metrics=fold_metrics, summary_metrics=summary)
