"""Data loading utilities for the dyslexia screening pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: tuple[str, str, str, str] = ("LX", "LY", "RX", "RY")


@dataclass(frozen=True)
class SubjectRecord:
    """Container for one subject recording."""

    subject_id: str
    label: int  # 0 = Control, 1 = Dyslexic
    frame: pd.DataFrame


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    clean = df.loc[:, REQUIRED_COLUMNS].copy()
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(axis=0, how="any")
    if clean.empty:
        raise ValueError(f"{path} has no valid rows after cleaning")
    return clean


def _iter_csv(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".csv")


def load_dataset(data_dir: str | Path) -> list[SubjectRecord]:
    """Load dataset from `data_dir/Control` and `data_dir/Dyslexic`."""
    data_dir = Path(data_dir)
    control_files = _iter_csv(data_dir / "Control")
    dyslexic_files = _iter_csv(data_dir / "Dyslexic")

    records: list[SubjectRecord] = []
    for path in control_files:
        sid = f"control::{path.stem}"
        records.append(SubjectRecord(subject_id=sid, label=0, frame=_read_csv(path)))

    for path in dyslexic_files:
        sid = f"dyslexic::{path.stem}"
        records.append(SubjectRecord(subject_id=sid, label=1, frame=_read_csv(path)))

    if not records:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. Expected Control/*.csv and Dyslexic/*.csv"
        )

    return records


def records_to_xy_groups(
    records: list[SubjectRecord],
) -> tuple[list[pd.DataFrame], np.ndarray, np.ndarray]:
    """Convert records to feature inputs, labels, and group ids."""
    x_frames = [r.frame for r in records]
    y = np.asarray([r.label for r in records], dtype=np.int64)
    groups = np.asarray([r.subject_id for r in records], dtype=object)
    return x_frames, y, groups
