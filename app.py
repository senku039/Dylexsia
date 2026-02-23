from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Dyslexia Detection Research Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --primary:#0B6E8A;
        --secondary:#2A9D8F;
        --bg:#F4FAFC;
        --card:#FFFFFF;
        --text:#103544;
    }
    .stApp {background: var(--bg); color: var(--text);}    
    .metric-card {
        background: var(--card);
        border: 1px solid #D7E8EE;
        border-left: 6px solid var(--primary);
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 4px 14px rgba(16, 53, 68, 0.06);
    }
    .section-card {
        background: var(--card);
        border: 1px solid #D7E8EE;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 14px rgba(16, 53, 68, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("🩺 Dyslexia Dashboard")
st.sidebar.caption("MSc Research • End-to-End Evolution")
st.sidebar.info(
    "Use the pages below to explore all 5 stages:\n"
    "1) Project Overview\n"
    "2) Early Work Analysis\n"
    "3) Binning Visualization\n"
    "4) STFT & Perceptron\n"
    "5) Production ML Evaluation"
)

st.title("Dyslexia Detection Research Dashboard")
st.markdown(
    """
<div class='section-card'>
This dashboard presents the complete R&D journey from early signal exploration to
production-grade ML evaluation. Use the sidebar to navigate across pages.
</div>
""",
    unsafe_allow_html=True,
)
