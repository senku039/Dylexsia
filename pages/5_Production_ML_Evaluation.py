from __future__ import annotations

import io
import subprocess
import sys

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


def _extract_table_block(text: str, header: str) -> str:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == header.strip():
            start = i + 1
            break
    if start is None:
        return ""

    block: list[str] = []
    for line in lines[start:]:
        if line.strip().startswith("===") and block:
            break
        if line.strip() == "" and block:
            break
        if line.strip() != "":
            block.append(line)
    return "\n".join(block)


def _parse_train_stdout(stdout: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    summary_block = _extract_table_block(stdout, "=== Unbiased Outer-Fold Metrics (mean ± 95% CI) ===")
    fold_block = _extract_table_block(stdout, "=== Fold Diagnostics ===")

    summary_df = None
    fold_df = None

    if summary_block:
        try:
            summary_df = pd.read_fwf(io.StringIO(summary_block))
            summary_df.columns = [str(c).strip() for c in summary_df.columns]
            if "metric" in summary_df.columns:
                summary_df = summary_df.set_index("metric")
        except Exception:
            summary_df = None

    if fold_block:
        try:
            fold_df = pd.read_fwf(io.StringIO(fold_block))
            fold_df.columns = [str(c).strip() for c in fold_df.columns]
        except Exception:
            fold_df = None

    return summary_df, fold_df


def _pretty_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    metric_names = {
        "accuracy": "Overall Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "sensitivity": "Sensitivity (Recall for Dyslexic)",
        "specificity": "Specificity (Recall for Control)",
        "f1": "F1 Score",
        "mcc": "Matthews Correlation (MCC)",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
    }

    df = summary_df.copy()
    for col in ["mean", "std", "ci95_low", "ci95_high"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    out = pd.DataFrame(index=df.index)
    out["Metric"] = [metric_names.get(idx, idx) for idx in df.index]
    out["Mean (%)"] = (df["mean"] * 100).round(2)
    out["95% CI (%)"] = (
        (df["ci95_low"] * 100).round(2).astype(str)
        + " to "
        + (df["ci95_high"] * 100).round(2).astype(str)
    )
    out["Std Dev (%)"] = (df["std"] * 100).round(2)
    return out.reset_index(drop=True)


def _pretty_fold_diag(fold_df: pd.DataFrame) -> pd.DataFrame:
    fd = fold_df.copy()
    for col in fd.columns:
        fd[col] = pd.to_numeric(fd[col], errors="ignore")

    rename_map = {
        "repeat": "Repeat",
        "fold": "Fold",
        "threshold": "Decision Threshold",
        "inner_best_roc_auc": "Best Inner ROC-AUC",
        "inner_best_score_roc_auc": "Best Inner ROC-AUC",
        "roc_auc": "Outer ROC-AUC",
        "pr_auc": "Outer PR-AUC",
        "balanced_accuracy": "Outer Balanced Accuracy",
    }
    fd = fd.rename(columns=rename_map)

    for col in ["Decision Threshold", "Best Inner ROC-AUC", "Outer ROC-AUC", "Outer PR-AUC", "Outer Balanced Accuracy"]:
        if col in fd.columns:
            fd[col] = pd.to_numeric(fd[col], errors="coerce").round(4)

    return fd


st.title("Production ML Evaluation")
st.caption("Run the production pipeline and view an easy-to-understand summary.")

smoke = st.toggle("Smoke-test mode", value=True)
model = st.selectbox("Model", ["logreg", "rf"], index=0)
splits = st.slider("CV splits", 2, 8, 5)
repeats = st.slider("CV repeats", 1, 5, 1)

if st.button("Run Evaluation", type="primary"):
    cmd = [
        sys.executable,
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

        summary_df, fold_df = _parse_train_stdout(proc.stdout)

        st.subheader("Simple Performance Summary")
        st.markdown(
            "- **Mean (%)**: average model performance across all outer folds.\n"
            "- **95% CI (%)**: expected range of performance stability.\n"
            "- **Std Dev (%)**: variability across folds (lower is better)."
        )

        if summary_df is not None and not summary_df.empty:
            pretty = _pretty_summary(summary_df)
            st.dataframe(pretty, width="stretch", hide_index=True)

            try:
                acc = float(summary_df.loc["accuracy", "mean"]) * 100
                bal = float(summary_df.loc["balanced_accuracy", "mean"]) * 100
                sens = float(summary_df.loc["sensitivity", "mean"]) * 100
                spec = float(summary_df.loc["specificity", "mean"]) * 100

                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(f"<div class='metric-card'><b>Accuracy</b><br><span style='font-size:1.35rem'>{acc:.2f}%</span></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='metric-card'><b>Balanced Accuracy</b><br><span style='font-size:1.35rem'>{bal:.2f}%</span></div>", unsafe_allow_html=True)
                c3.markdown(f"<div class='metric-card'><b>Sensitivity</b><br><span style='font-size:1.35rem'>{sens:.2f}%</span></div>", unsafe_allow_html=True)
                c4.markdown(f"<div class='metric-card'><b>Specificity</b><br><span style='font-size:1.35rem'>{spec:.2f}%</span></div>", unsafe_allow_html=True)
            except Exception:
                pass
        else:
            st.info("Could not parse summary table from train output.")

        st.subheader("Fold-by-Fold Diagnostics")
        st.markdown("Each row is one outer-fold run. Use this table to inspect consistency.")
        if fold_df is not None and not fold_df.empty:
            fold_pretty = _pretty_fold_diag(fold_df)
            st.dataframe(fold_pretty, width="stretch", hide_index=True)

            csv_data = fold_pretty.to_csv(index=False)
            st.download_button(
                "Download Fold Diagnostics CSV",
                data=csv_data,
                file_name="train_fold_diagnostics.csv",
                mime="text/csv",
            )
        else:
            st.info("Could not parse fold diagnostics table from train output.")

        with st.expander("Raw Console Output"):
            st.code(proc.stdout)

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
    col1.plotly_chart(roc_fig, width="stretch")

    cm_df = pd.DataFrame(cm, index=["True Control", "True Dyslexic"], columns=["Pred Control", "Pred Dyslexic"])
    cm_fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Teal", title="Confusion Matrix")
    cm_fig.update_layout(template="plotly_white", height=400)
    col2.plotly_chart(cm_fig, width="stretch")

    diag = pd.DataFrame({"id": idte, "y_true": yte, "y_pred": pred, "prob_dyslexic": prob})
    st.subheader("Fold Diagnostics (proxy holdout diagnostics)")
    st.dataframe(diag, width="stretch")

    csv_buf = io.StringIO()
    diag.to_csv(csv_buf, index=False)
    st.download_button("Download Metrics CSV", data=csv_buf.getvalue(), file_name="production_ml_diagnostics.csv", mime="text/csv")
else:
    st.info("Control/Dyslexic data not found; KPI and diagnostics preview unavailable.")
