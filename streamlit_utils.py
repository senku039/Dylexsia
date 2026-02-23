from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


DATA_DIR = Path("Data")


def load_subject_table(group: str) -> dict[str, pd.DataFrame]:
    folder = DATA_DIR / group
    out: dict[str, pd.DataFrame] = {}
    if not folder.exists():
        return out

    for p in sorted(folder.glob("*.csv")):
        try:
            df = pd.read_csv(p)
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])
            cols = [c for c in ["LX", "LY", "RX", "RY"] if c in df.columns]
            if len(cols) < 4:
                continue
            out[p.stem] = df[["LX", "LY", "RX", "RY"]].copy()
        except Exception:
            continue
    return out


def series_figure(df: pd.DataFrame, title: str) -> go.Figure:
    plot_df = df.copy().reset_index().rename(columns={"index": "t"})
    plot_df = plot_df.melt(id_vars="t", var_name="channel", value_name="value")
    fig = px.line(plot_df, x="t", y="value", color="channel", title=title)
    fig.update_layout(template="plotly_white", legend_title_text="Channel", height=420)
    return fig


def fft_figure(df: pd.DataFrame, title: str) -> go.Figure:
    numeric_df = df.select_dtypes(include=["number"]).copy()

    if {"LX", "RX", "LY", "RY"}.issubset(numeric_df.columns):
        x = numeric_df[["LX", "RX"]].mean(axis=1)
        y = numeric_df[["LY", "RY"]].mean(axis=1)
        z = x + 1j * y
    elif len(numeric_df.columns) > 0:
        z = numeric_df.mean(axis=1)
    else:
        z = pd.Series([], dtype=float)

    z = pd.to_numeric(z, errors="coerce")
    z = z.dropna()
    z = np.asarray(z, dtype=float)
    z = z.flatten()

    if z.size < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough valid numeric data for FFT",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(template="plotly_white", title=title, height=420)
        return fig

    try:
        spec = np.abs(np.fft.rfft(z))
        f = np.arange(len(spec))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f, y=spec, mode="lines", name="|FFT|"))
        fig.update_layout(
            template="plotly_white",
            title=title,
            xaxis_title="Frequency Bin",
            yaxis_title="Magnitude",
            height=420,
        )
        return fig
    except Exception as exc:
        fig = go.Figure()
        fig.add_annotation(
            text=f"FFT failed: {exc}",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(template="plotly_white", title=title, height=420)
        return fig


def stft_heatmap(df: pd.DataFrame, title: str) -> go.Figure:
    from scipy.signal import stft

    x = df[["LX", "RX"]].mean(axis=1).to_numpy(dtype=float)
    f, t, zxx = stft(x, nperseg=min(128, len(x)), noverlap=min(64, max(0, len(x)-1)))
    power = np.abs(zxx) ** 2

    fig = px.imshow(power, aspect="auto", origin="lower", labels={"x": "Time Bin", "y": "Frequency Bin", "color": "Power"}, title=title)
    fig.update_layout(template="plotly_white", height=450)
    return fig
