from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

from streamlit_utils import load_subject_table, stft_heatmap

st.title("STFT & Perceptron")

cohort = st.selectbox("Cohort", ["Control", "Dyslexic"], index=0)
subjects = load_subject_table(cohort)
if not subjects:
    st.warning("No data available.")
    st.stop()

sid = st.selectbox("Candidate", list(subjects.keys()))
df = subjects[sid]
st.plotly_chart(stft_heatmap(df, f"STFT Spectrogram • {cohort} • {sid}"), use_container_width=True)

st.subheader("Perceptron Comparison (Stage-4 style linear classifier)")
control = load_subject_table("Control")
dys = load_subject_table("Dyslexic")

rows = []
for k, d in control.items():
    x = d[["LX", "RX"]].mean(axis=1).to_numpy(dtype=float)
    y = d[["LY", "RY"]].mean(axis=1).to_numpy(dtype=float)
    rows.append((k, 0, np.array([x.mean(), x.std(), y.mean(), y.std(), len(d)], dtype=float)))
for k, d in dys.items():
    x = d[["LX", "RX"]].mean(axis=1).to_numpy(dtype=float)
    y = d[["LY", "RY"]].mean(axis=1).to_numpy(dtype=float)
    rows.append((k, 1, np.array([x.mean(), x.std(), y.mean(), y.std(), len(d)], dtype=float)))

ids = [r[0] for r in rows]
y = np.array([r[1] for r in rows], dtype=int)
X = np.vstack([r[2] for r in rows])

Xtr, Xte, ytr, yte, idtr, idte = train_test_split(X, y, ids, test_size=0.3, random_state=42, stratify=y)
clf = Perceptron(random_state=42, max_iter=5000)
clf.fit(Xtr, ytr)
yp = clf.predict(Xte)
acc = float((yp == yte).mean())

out = pd.DataFrame({"id": idte, "true": yte, "pred": yp})
out["status"] = np.where(out["true"] == out["pred"], "Correct", "Incorrect")
fig = px.histogram(out, x="status", color="status", title=f"Perceptron Holdout Results • Accuracy={acc:.3f}")
fig.update_layout(template="plotly_white", height=380)
st.plotly_chart(fig, use_container_width=True)
