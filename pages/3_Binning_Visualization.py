from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from streamlit_utils import fft_figure, load_subject_table


def _safe_binned_fft_vector(df: pd.DataFrame, bins: int) -> np.ndarray | None:
    df_numeric = df.select_dtypes(include=["number"])
    if df_numeric.shape[1] == 0:
        return None

    z = df_numeric.values.flatten()
    z = z.astype(float)
    z = z[~np.isnan(z)]

    if len(z) < 2:
        return None

    try:
        spec = np.abs(np.fft.rfft(z))
    except Exception:
        return None

    chunk = np.array_split(spec, bins)
    return np.array([float(np.sum(c)) for c in chunk], dtype=float)


st.title("Binning Visualization")

cohort = st.selectbox("Cohort", ["Control", "Dyslexic"], index=0)
subjects = load_subject_table(cohort)
if not subjects:
    st.warning("No data found.")
    st.stop()

sid = st.selectbox("Candidate", list(subjects.keys()))
df = subjects[sid]

st.plotly_chart(fft_figure(df, f"FFT Spectrum • {cohort} • {sid}"), width="stretch")

st.subheader("Approximate Binning + KMeans Cluster View")
all_control = load_subject_table("Control")
all_dys = load_subject_table("Dyslexic")
all_data = [(k, 0, v) for k, v in all_control.items()] + [(k, 1, v) for k, v in all_dys.items()]

bins = st.slider("Number of FFT bins", 16, 128, 48, 8)

vectors = []
labels = []
ids = []
for sid, y, d in all_data:
    vec = _safe_binned_fft_vector(d, bins=bins)
    if vec is None:
        continue
    vectors.append(vec)
    labels.append("Control" if y == 0 else "Dyslexic")
    ids.append(sid)

if len(vectors) < 3:
    st.warning("Not enough valid numeric data to compute binned FFT cluster view.")
    st.stop()

X = np.vstack(vectors)
X2 = PCA(n_components=2, random_state=42).fit_transform(X)
km = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = km.fit_predict(X)

plot_df = pd.DataFrame({
    "pc1": X2[:, 0],
    "pc2": X2[:, 1],
    "label": labels,
    "cluster": clusters.astype(str),
    "id": ids,
})
fig = px.scatter(plot_df, x="pc1", y="pc2", color="cluster", symbol="label", hover_name="id", title="KMeans on Binned FFT Features")
fig.update_layout(template="plotly_white", height=500)
st.plotly_chart(fig, width="stretch")
