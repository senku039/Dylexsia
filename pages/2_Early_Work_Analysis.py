from __future__ import annotations

import streamlit as st

from streamlit_utils import load_subject_table, series_figure

st.title("Early Work Analysis")

st.caption("Interactive raw eye-tracking visualization for exploratory analysis.")

cohort = st.selectbox("Select Cohort", ["Control", "Dyslexic"], index=0)
data = load_subject_table(cohort)

if not data:
    st.warning("No valid CSV data found for this cohort.")
    st.stop()

subject = st.selectbox("Select Candidate", list(data.keys()))
df = data[subject]

st.plotly_chart(series_figure(df, f"Raw Eye-Tracking Signals • {cohort} • {subject}"), use_container_width=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Samples", len(df))
c2.metric("LX mean", f"{df['LX'].mean():.3f}")
c3.metric("LY mean", f"{df['LY'].mean():.3f}")
c4.metric("RX-RY corr", f"{df['RX'].corr(df['RY']):.3f}")
