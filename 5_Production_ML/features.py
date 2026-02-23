"""Feature engineering for variable-length eye-tracking sequences."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import stft


class EyeTrackingFeatureExtractor:
    """Extract robust fixed-length features from LX/LY/RX/RY trajectories."""

    def __init__(self, stft_nperseg: int = 128, stft_noverlap: int = 64) -> None:
        self.stft_nperseg = stft_nperseg
        self.stft_noverlap = stft_noverlap

    @staticmethod
    def _xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = frame[["LX", "RX"]].mean(axis=1).to_numpy(dtype=float)
        y = frame[["LY", "RY"]].mean(axis=1).to_numpy(dtype=float)
        return x, y

    @staticmethod
    def _stats(sig: np.ndarray, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_mean": float(np.mean(sig)),
            f"{prefix}_std": float(np.std(sig)),
            f"{prefix}_median": float(np.median(sig)),
            f"{prefix}_q25": float(np.quantile(sig, 0.25)),
            f"{prefix}_q75": float(np.quantile(sig, 0.75)),
            f"{prefix}_min": float(np.min(sig)),
            f"{prefix}_max": float(np.max(sig)),
        }

    @staticmethod
    def _fft_features(sig: np.ndarray, prefix: str) -> dict[str, float]:
        centered = sig - np.mean(sig)
        psd = np.abs(np.fft.rfft(centered)) ** 2
        total = float(psd.sum()) + 1e-12

        n = len(psd)
        i1, i2, i3 = max(1, n // 4), max(2, n // 2), max(3, 3 * n // 4)
        q1 = float(psd[:i1].sum() / total)
        q2 = float(psd[i1:i2].sum() / total)
        q3 = float(psd[i2:i3].sum() / total)
        q4 = float(psd[i3:].sum() / total)

        probs = psd / total
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        peak_bin = float(np.argmax(psd))
        return {
            f"{prefix}_fft_q1": q1,
            f"{prefix}_fft_q2": q2,
            f"{prefix}_fft_q3": q3,
            f"{prefix}_fft_q4": q4,
            f"{prefix}_fft_entropy": entropy,
            f"{prefix}_fft_peak_bin": peak_bin,
        }

    def _stft_features(self, sig: np.ndarray, prefix: str) -> dict[str, float]:
        nperseg = min(self.stft_nperseg, len(sig))
        noverlap = min(self.stft_noverlap, max(0, nperseg - 1))
        _, _, zxx = stft(sig, nperseg=nperseg, noverlap=noverlap)

        power = np.abs(zxx) ** 2
        band = power.mean(axis=1)
        k = max(1, len(band) // 4)
        return {
            f"{prefix}_stft_low": float(np.mean(band[:k])),
            f"{prefix}_stft_mid_low": float(np.mean(band[k : 2 * k])),
            f"{prefix}_stft_mid_high": float(np.mean(band[2 * k : 3 * k])),
            f"{prefix}_stft_high": float(np.mean(band[3 * k :])),
            f"{prefix}_stft_global_std": float(np.std(power)),
        }

    def transform_one(self, frame: pd.DataFrame) -> dict[str, float]:
        x, y = self._xy(frame)

        pos = np.sqrt(x**2 + y**2)
        vx = np.diff(x, prepend=x[0])
        vy = np.diff(y, prepend=y[0])
        speed = np.sqrt(vx**2 + vy**2)

        ax = np.diff(vx, prepend=vx[0])
        ay = np.diff(vy, prepend=vy[0])
        accel = np.sqrt(ax**2 + ay**2)

        out: dict[str, float] = {"seq_len": float(len(frame))}
        out.update(self._stats(x, "x"))
        out.update(self._stats(y, "y"))
        out.update(self._stats(pos, "pos"))
        out.update(self._stats(speed, "speed"))
        out.update(self._stats(accel, "accel"))

        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            out["xy_corr"] = 0.0
        else:
            out["xy_corr"] = float(np.corrcoef(x, y)[0, 1])

        out.update(self._fft_features(x, "x"))
        out.update(self._fft_features(y, "y"))
        out.update(self._stft_features(x, "x"))
        out.update(self._stft_features(y, "y"))
        return out

    def transform(self, frames: list[pd.DataFrame]) -> pd.DataFrame:
        rows = [self.transform_one(f) for f in frames]
        feat = pd.DataFrame(rows)
        return feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
