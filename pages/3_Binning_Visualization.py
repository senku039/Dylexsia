from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from streamlit_utils import fft_figure, load_subject_table

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
    x = d[["LX", "RX"]].mean(axis=1).to_numpy(dtype=float)
    yv = d[["LY", "RY"]].mean(axis=1).to_numpy(dtype=float)
    z = x + 1j * yv
    spec = np.abs(np.fft.rfft(z))
    chunk = np.array_split(spec, bins)
    vec = np.array([float(np.sum(c)) for c in chunk], dtype=float)
    vectors.append(vec)
    labels.append("Control" if y == 0 else "Dyslexic")
    ids.append(sid)

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
