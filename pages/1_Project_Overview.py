from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

st.title("Project Overview")

st.markdown("""
### Research Motivation
Dyslexia screening based on eye-tracking can provide scalable and objective pre-assessment support.
This MSc project explores progressive modeling approaches from exploratory preprocessing to production-grade ML.
""")

st.markdown("""
### Dataset Description
- Two cohorts: **Control** and **Dyslexic**
- Channels per time point: **LX, LY, RX, RY**
- Variable-length recordings requiring equalization/feature engineering
""")

st.markdown("""
### Clinical Objective
Build a robust triage-support model for dyslexia screening that is:
- interpretable,
- reproducible,
- evaluated with leakage-resistant grouped validation.
""")

st.subheader("System Architecture Diagram")

fig = go.Figure()
nodes = {
    "Data": (0, 0),
    "Early Work": (1, 1),
    "Binning": (2, 1),
    "STFT/Perceptron": (3, 1),
    "Production ML": (4, 1),
    "Clinical Reporting": (5, 0),
}

edges = [
    ("Data", "Early Work"),
    ("Early Work", "Binning"),
    ("Binning", "STFT/Perceptron"),
    ("STFT/Perceptron", "Production ML"),
    ("Production ML", "Clinical Reporting"),
]

for a, b in edges:
    xa, ya = nodes[a]
    xb, yb = nodes[b]
    fig.add_shape(type="line", x0=xa, y0=ya, x1=xb, y1=yb, line=dict(color="#0B6E8A", width=3))

for name, (x, y) in nodes.items():
    fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", marker=dict(size=32, color="#2A9D8F"), text=[name], textposition="bottom center", showlegend=False))

fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)
