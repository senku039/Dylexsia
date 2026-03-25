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
st.sidebar.caption("MSc Research • Simplified Home")
st.sidebar.info(
    "Use the remaining analysis pages from the sidebar.\n\n"
    "Quick info cards are now shown as popups on this home screen."
)

st.title("Dyslexia Detection Research Dashboard")
st.markdown(
    """
<div class='section-card'>
Quick educational information is available below as popup cards.
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)

with c1:
    with st.popover("🧠 What is Dyslexia?"):
        st.markdown(
            """
Dyslexia is a language-based learning difference that can affect reading fluency,
spelling, decoding and writing patterns. It is not linked to intelligence.
"""
        )

with c2:
    with st.popover("✅ Common Signs"):
        st.markdown(
            """
- Slow/effortful reading
- Difficulty mapping sounds to letters
- Spelling inconsistency
- Skipping/re-reading words while reading
"""
        )

with c3:
    with st.popover("🤝 Support Paths"):
        st.markdown(
            """
- Structured literacy interventions
- School accommodations
- Home reading routines
- Assistive technology support
"""
        )

st.divider()
st.subheader("Use Our Tools")
st.write(
    "Open the remaining pages from the sidebar for STFT/Perceptron analysis and "
    "Production ML evaluation."
)
