"""
app.py
------
Main application entry point for the IPL Win Predictor & Sports Analytics Suite.
"""

import streamlit as st
import pandas as pd
from src.ui_helpers import load_custom_css, render_sidebar_brand, render_kpi_card
from src.data_loader import load_matches_data, load_deliveries_data
from src.analytics import get_overall_kpis

# Page configuration
st.set_page_config(
    page_title="IPL Win Predictor | ML Analytics Platform",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styles and sidebar branding
load_custom_css()
render_sidebar_brand()

# Load datasets
matches_df = load_matches_data()
deliveries_df = load_deliveries_data()
kpi_data = get_overall_kpis(matches_df, deliveries_df)

# Sidebar metadata info
st.sidebar.markdown(
    """
    <div style="background: rgba(15, 23, 42, 0.6); padding: 1rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); margin-top: 1rem;">
        <div style="font-size: 0.75rem; text-transform: uppercase; color: #94A3B8; font-weight: 700; letter-spacing: 0.05em;">Pipeline Status</div>
        <div style="color: #10B981; font-weight: 700; font-size: 0.9rem; margin-top: 0.2rem;">● pipe.pkl Active</div>
        <div style="font-size: 0.8rem; color: #CBD5E1; margin-top: 0.4rem;">
            Logistic Regression Pipeline<br>
            9 Input Features Trained<br>
            Predict Proba: Supported
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Hero Header
st.markdown(
    """
    <div style="padding: 1.5rem 0 1rem 0;">
        <span class="badge-pill badge-orange">Indian Premier League • ML Analytics</span>
        <h1 style="margin: 0.5rem 0 0.3rem 0; font-size: 2.8rem;">
            IPL <span class="gradient-title">Win Predictor</span> & Analytics
        </h1>
        <p style="color: #94A3B8; font-size: 1.1rem; max-width: 850px; line-height: 1.6;">
            A high-performance machine learning platform powered by historical ball-by-ball IPL match data. 
            Forecast in-play 2nd-innings chase outcomes with exact probabilistic modeling, explore franchise metrics, 
            and analyze tactical cricket dynamics.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# KPI Row
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    render_kpi_card("Total Matches", f"{kpi_data['total_matches']:,}", "Historical Records", "2008-2024+")
with col2:
    render_kpi_card("IPL Seasons", f"{kpi_data['total_seasons']}", kpi_data['seasons_range'], "Editions")
with col3:
    render_kpi_card("Franchises", f"{kpi_data['total_teams']}", "Active & Historic", "Teams")
with col4:
    render_kpi_card("Total Deliveries", f"{len(deliveries_df):,}", "Ball-by-Ball Data", "Balls")
with col5:
    render_kpi_card("Total Runs", f"{kpi_data['total_runs']:,}", "All-Time Scored", "Runs")

st.markdown("<br>", unsafe_allow_html=True)

# Main Navigation Action Cards
st.markdown("### ⚡ Quick Navigation")
nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    st.markdown(
        """
        <div class="glass-card" style="height: 230px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <span style="font-size: 2rem;">🔮</span>
                <h3 style="margin: 0.5rem 0 0.3rem 0; font-size: 1.3rem; color: #00D2FF !important;">Match Win Predictor</h3>
                <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.5;">
                    Live 2nd-innings chase win probability calculator. Uses real-time CRR, RRR, wickets left, and pitch venue.
                </p>
            </div>
            <div style="font-size: 0.85rem; color: #8B5CF6; font-weight: 600;">Go to: 2_Match_Prediction ➔</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with nav_col2:
    st.markdown(
        """
        <div class="glass-card" style="height: 230px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <span style="font-size: 2rem;">🏆</span>
                <h3 style="margin: 0.5rem 0 0.3rem 0; font-size: 1.3rem; color: #FF6B00 !important;">Team Analytics</h3>
                <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.5;">
                    Franchise deep-dives, head-to-head match histories, venue strengths, and defending vs chasing splits.
                </p>
            </div>
            <div style="font-size: 0.85rem; color: #FF6B00; font-weight: 600;">Go to: 3_Team_Analytics ➔</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with nav_col3:
    st.markdown(
        """
        <div class="glass-card" style="height: 230px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <span style="font-size: 2rem;">📊</span>
                <h3 style="margin: 0.5rem 0 0.3rem 0; font-size: 1.3rem; color: #10B981 !important;">League Insights</h3>
                <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.5;">
                    Comprehensive IPL visualizations: toss impact, venue averages, season evolutions, and victory margins.
                </p>
            </div>
            <div style="font-size: 0.85rem; color: #10B981; font-weight: 600;">Go to: 4_Data_Analytics ➔</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# System Architecture Highlights
st.markdown("### ⚙️ Pipeline Highlights")
arch_col1, arch_col2 = st.columns([1.5, 1])

with arch_col1:
    st.markdown(
        """
        <div class="glass-card">
            <h4 style="margin-top: 0; color: #FFFFFF !important;">Active Machine Learning Architecture</h4>
            <p style="color: #94A3B8; font-size: 0.95rem; line-height: 1.6;">
                The prediction system is integrated with the pre-trained <code>pipe.pkl</code> serialized pipeline.
                It incorporates a two-stage column transformation process:
            </p>
            <ul style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.8;">
                <li><b>Categorical Sub-Pipeline:</b> Imputation (<code>Unknown</code>) followed by <code>OneHotEncoder(drop='first', handle_unknown='ignore')</code> across <code>['batting_team', 'bowling_team', 'city']</code>.</li>
                <li><b>Numerical Sub-Pipeline:</b> Median imputation for dynamic cricket metrics: <code>['runs_left', 'balls_left', 'wickets_left', 'runs_total_x', 'crr', 'rrr']</code>.</li>
                <li><b>Final Estimator:</b> <code>LogisticRegression(solver='liblinear')</code> generating calibrated win/loss probabilities.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

with arch_col2:
    st.markdown(
        """
        <div class="glass-card">
            <h4 style="margin-top: 0; color: #FFFFFF !important;">Dataset Specifications</h4>
            <div style="display: flex; flex-direction: column; gap: 0.6rem; margin-top: 0.8rem;">
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 0.4rem;">
                    <span style="color: #94A3B8;">Matches Dataset:</span>
                    <span style="color: #00D2FF; font-weight: 600;">1,243 rows</span>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 0.4rem;">
                    <span style="color: #94A3B8;">Deliveries Dataset:</span>
                    <span style="color: #00D2FF; font-weight: 600;">295,732 deliveries</span>
                </div>
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 0.4rem;">
                    <span style="color: #94A3B8;">Unique Venues:</span>
                    <span style="color: #00D2FF; font-weight: 600;">50+ stadiums</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #94A3B8;">Trained Target:</span>
                    <span style="color: #10B981; font-weight: 600;">Chasing Team Win (1/0)</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
