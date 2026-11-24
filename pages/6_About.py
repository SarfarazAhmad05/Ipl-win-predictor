"""
pages/6_About.py
----------------
About Project: Technical documentation, problem statement, ML workflow,
dataset origins, software architecture, limitations, and future enhancements.
"""

import streamlit as st
from src.ui_helpers import load_custom_css, render_sidebar_brand

st.set_page_config(page_title="About Project | IPL Win Predictor", page_icon="ℹ️", layout="wide")
load_custom_css()
render_sidebar_brand()

st.markdown(
    """
    <div style="padding: 1rem 0;">
        <span class="badge-pill badge-blue">Page 6 • Technical Documentation</span>
        <h1 style="margin: 0.4rem 0;">About IPL Win Predictor</h1>
        <p style="color: #94A3B8; font-size: 1.05rem;">
            Architecture, engineering lifecycle, dataset specifications, limitations, and future development scope.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.markdown("### 🎯 Problem Statement & Objectives")
    st.markdown(
        """
        <div class="glass-card">
            <p style="color: #CBD5E1; font-size: 0.95rem; line-height: 1.7;">
                Twenty20 cricket is an inherently volatile, high-paced sporting format where match momentum fluctuates 
                dramatically from delivery to delivery. Static pre-match models fail to reflect in-game momentum shifts 
                such as rapid run accumulation, clumping of wickets, or escalating required run rates in second-innings chases.
            </p>
            <h5 style="color: #00D2FF; margin-top: 1rem;">Core Project Objectives:</h5>
            <ul style="color: #94A3B8; font-size: 0.9rem; line-height: 1.8;">
                <li><b>In-Play Dynamic Forecasting:</b> Generate probabilistic win trajectories ball-by-ball during the 2nd innings chase.</li>
                <li><b>Zero Data Leakage:</b> Strictly isolate preprocessing transformers using a unified Scikit-Learn <code>Pipeline</code>.</li>
                <li><b>Deterministic Execution:</b> Integrate an existing pre-trained serialized model (<code>pipe.pkl</code>) without retraining or altering weights.</li>
                <li><b>Sports Analytics Dashboard:</b> Provide interactive macro and micro analytics across 1,243 historical IPL matches.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🔄 Machine Learning Workflow")
    st.markdown(
        """
        <div class="glass-card">
            <ol style="color: #CBD5E1; font-size: 0.9rem; line-height: 2;">
                <li><b>Data Acquisition:</b> Ingestion of 1,243 IPL matches and 295,732 ball-by-ball delivery events (2008–Present).</li>
                <li><b>Target Engineering:</b> First-innings cumulative totals aggregated to establish target scores; 2nd innings filtered to create binary chase outcomes (<code>result = 1</code> if chasing team wins, else <code>0</code>).</li>
                <li><b>Dynamic Feature Extraction:</b> Derivation of <code>runs_left</code>, <code>balls_left</code>, <code>wickets_left</code>, <code>crr</code>, and <code>rrr</code>.</li>
                <li><b>Pipeline Encapsulation:</b> <code>ColumnTransformer</code> comprising Median Imputation for numeric columns and One-Hot Encoding for categorical features.</li>
                <li><b>Probabilistic Modeling:</b> Logistic Regression with <code>liblinear</code> solver optimizing log-loss to generate calibrated sigmoid outputs.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_right:
    st.markdown("### 🛠️ Technology Stack")
    st.markdown(
        """
        <div class="glass-card">
            <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="color: #94A3B8; font-weight: 600;">Core Language:</span>
                    <span class="badge-pill badge-purple">Python 3.10+</span>
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="color: #94A3B8; font-weight: 600;">Interactive Web App:</span>
                    <span class="badge-pill badge-orange">Streamlit 1.40+</span>
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="color: #94A3B8; font-weight: 600;">Machine Learning:</span>
                    <span class="badge-pill badge-blue">Scikit-Learn 1.5+</span>
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="color: #94A3B8; font-weight: 600;">Data Manipulation:</span>
                    <span class="badge-pill badge-purple">Pandas & NumPy</span>
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="color: #94A3B8; font-weight: 600;">Interactive Graphics:</span>
                    <span class="badge-pill badge-orange">Plotly Graph Objects</span>
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="color: #94A3B8; font-weight: 600;">Model Persistence:</span>
                    <span class="badge-pill badge-blue">Joblib / Pickle</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### ⚠️ Limitations & Future Scope")
    st.markdown(
        """
        <div class="glass-card">
            <h5 style="color: #F59E0B; margin-top: 0;">Current Model Boundaries:</h5>
            <ul style="color: #94A3B8; font-size: 0.85rem; line-height: 1.6;">
                <li>Individual batter and bowler matchups (e.g. death-over specialists, powerplay hitters) are aggregated at the franchise level.</li>
                <li>Weather conditions, dew factor, and pitch degradation over the match are represented solely by host city proxies.</li>
                <li>Linear decision boundary (Logistic Regression) may underfit complex nonlinear phase transitions.</li>
            </ul>
            <h5 style="color: #10B981; margin-top: 1rem;">Future Roadmap:</h5>
            <ul style="color: #94A3B8; font-size: 0.85rem; line-height: 1.6;">
                <li>Incorporation of real-time ball tracking and player form embeddings.</li>
                <li>Gradient Boosting (XGBoost/LightGBM) & Recurrent Neural Networks (LSTM/GRU) for temporal ball-sequence modeling.</li>
                <li>Live ball-by-ball API stream integration via Cricinfo / Cricbuzz APIs.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
