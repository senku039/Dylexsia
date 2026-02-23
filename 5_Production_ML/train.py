"""CLI entrypoint for production-grade dyslexia model evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_dataset, records_to_xy_groups
from features import EyeTrackingFeatureExtractor
from model import evaluate_grouped_nested_cv


def _make_synthetic_data(
    n_subjects: int = 20,
    repeats_per_subject: int = 2,
    seq_len: int = 1000,
    random_state: int = 42,
) -> tuple[list[pd.DataFrame], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)

    frames: list[pd.DataFrame] = []
    labels: list[int] = []
    groups: list[str] = []

    for sid in range(n_subjects):
        label = 1 if sid >= n_subjects // 2 else 0
        for rep in range(repeats_per_subject):
            noise = rng.normal(0.0, 1.0, size=(seq_len, 4))
            drift = np.linspace(0.0, 0.03 if label == 1 else 0.01, seq_len)[:, None]
            signal = noise + drift + (0.15 if label == 1 else 0.0)
            frame = pd.DataFrame(signal, columns=["LX", "LY", "RX", "RY"])

            frames.append(frame)
            labels.append(label)
            groups.append(f"subject_{sid:03d}")

    return frames, np.asarray(labels, dtype=np.int64), np.asarray(groups, dtype=object)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dyslexia detection grouped nested CV evaluator")
    parser.add_argument("--data-dir", type=Path, default=Path("Data"), help="Root data directory")
    parser.add_argument("--model", choices=["logreg", "rf"], default="logreg", help="Model type")
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic end-to-end smoke test")
    parser.add_argument("--splits", type=int, default=5, help="StratifiedGroupKFold splits")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeated grouped CV runs")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--calibration",
        choices=["platt", "isotonic", "none"],
        default="platt",
        help="Probability calibration strategy",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.smoke_test:
        frames, y, groups = _make_synthetic_data(random_state=args.random_state)
    else:
        records = load_dataset(args.data_dir)
        frames, y, groups = records_to_xy_groups(records)

    extractor = EyeTrackingFeatureExtractor()
    x = extractor.transform(frames)

    result = evaluate_grouped_nested_cv(
        x=x,
        y=y,
        groups=groups,
        model_name=args.model,
        splits=args.splits,
        repeats=args.repeats,
        random_state=args.random_state,
        calibration=args.calibration,
    )

    print("\n=== Unbiased Outer-Fold Metrics (mean ± 95% CI) ===")
    print(result.summary_metrics.to_string(float_format=lambda v: f"{v:0.4f}"))

    print("\n=== Fold Diagnostics ===")
    print(
        result.fold_metrics[
            ["repeat", "fold", "threshold", "inner_best_roc_auc", "roc_auc", "pr_auc", "balanced_accuracy"]
        ].to_string(index=False, float_format=lambda v: f"{v:0.4f}")
    )


if __name__ == "__main__":
    main()
