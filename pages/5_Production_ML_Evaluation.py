from __future__ import annotations

import io
import subprocess

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from streamlit_utils import load_subject_table

st.title("Production ML Evaluation")

smoke = st.toggle("Smoke-test mode", value=True)
model = st.selectbox("Model", ["logreg", "rf"], index=0)
splits = st.slider("CV splits", 2, 8, 5)
repeats = st.slider("CV repeats", 1, 5, 1)

if st.button("Run Evaluation", type="primary"):
    cmd = [
        "python",
        "5_Production_ML/train.py",
        "--model",
        model,
        "--splits",
        str(splits),
        "--repeats",
        str(repeats),
        "--random-state",
        "42",
    ]
    if smoke:
        cmd.append("--smoke-test")
    else:
        cmd.extend(["--data-dir", "Data"])

    with st.spinner("Running production evaluation..."):
        proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        st.error("Evaluation command failed.")
        st.code(proc.stderr or proc.stdout)
    else:
        st.success("Evaluation completed")
        st.text_area("Console Summary", proc.stdout, height=220)

# Dashboard visuals from lightweight local estimate for clean UI
control = load_subject_table("Control")
dys = load_subject_table("Dyslexic")
if control and dys:
    rows = []
    for k, d in control.items():
        x = d[["LX", "RX"]].mean(axis=1).to_numpy(dtype=float)
        y = d[["LY", "RY"]].mean(axis=1).to_numpy(dtype=float)
        rows.append((k, 0, np.array([x.mean(), x.std(), y.mean(), y.std(), len(d)])))
    for k, d in dys.items():
        x = d[["LX", "RX"]].mean(axis=1).to_numpy(dtype=float)
        y = d[["LY", "RY"]].mean(axis=1).to_numpy(dtype=float)
        rows.append((k, 1, np.array([x.mean(), x.std(), y.mean(), y.std(), len(d)])))

    ids = [r[0] for r in rows]
    y = np.array([r[1] for r in rows], dtype=int)
    X = np.vstack([r[2] for r in rows])

    Xtr, Xte, ytr, yte, idtr, idte = train_test_split(X, y, ids, test_size=0.3, random_state=42, stratify=y)
    clf = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))])
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)

    cm = confusion_matrix(yte, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    bal = (sens + spec) / 2

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><b>Accuracy</b><br><span style='font-size:1.4rem'>{acc:.3f}</span></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><b>Balanced Accuracy</b><br><span style='font-size:1.4rem'>{bal:.3f}</span></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><b>Sensitivity</b><br><span style='font-size:1.4rem'>{sens:.3f}</span></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><b>Specificity</b><br><span style='font-size:1.4rem'>{spec:.3f}</span></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    fpr, tpr, _ = roc_curve(yte, prob)
    roc_auc = auc(fpr, tpr)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC AUC={roc_auc:.3f}"))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))
    roc_fig.update_layout(template="plotly_white", title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", height=400)
    col1.plotly_chart(roc_fig, use_container_width=True)

    cm_df = pd.DataFrame(cm, index=["True Control", "True Dyslexic"], columns=["Pred Control", "Pred Dyslexic"])
    cm_fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Teal", title="Confusion Matrix")
    cm_fig.update_layout(template="plotly_white", height=400)
    col2.plotly_chart(cm_fig, use_container_width=True)

    diag = pd.DataFrame({"id": idte, "y_true": yte, "y_pred": pred, "prob_dyslexic": prob})
    st.subheader("Fold Diagnostics (proxy holdout diagnostics)")
    st.dataframe(diag, use_container_width=True)

    csv_buf = io.StringIO()
    diag.to_csv(csv_buf, index=False)
    st.download_button("Download Metrics CSV", data=csv_buf.getvalue(), file_name="production_ml_diagnostics.csv", mime="text/csv")
else:
    st.info("Control/Dyslexic data not found; KPI and diagnostics preview unavailable.")
