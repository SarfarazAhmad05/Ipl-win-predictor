"""
pages/2_Match_Prediction.py
---------------------------
Core Feature: In-play IPL 2nd-innings match win prediction.
Integrates directly with the trained scikit-learn pipeline (pipe.pkl).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.ui_helpers import load_custom_css, render_sidebar_brand, apply_plotly_theme, render_kpi_card
from src.prediction import (
    predict_match_outcome,
    SUPPORTED_TEAMS,
    SUPPORTED_CITIES,
    get_pipeline
)
from src.data_loader import load_matches_data

st.set_page_config(page_title="Match Prediction | IPL Win Predictor", page_icon="⚡", layout="wide")
load_custom_css()
render_sidebar_brand()

# Ensure model is loadable
try:
    pipe = get_pipeline()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_err = str(e)

matches_df = load_matches_data()

# Header
st.markdown(
    """
    <div style="padding: 1rem 0;">
        <span class="badge-pill badge-orange">Page 2 • Core ML Engine</span>
        <h1 style="margin: 0.4rem 0;">IPL Match Win Probability Predictor</h1>
        <p style="color: #94A3B8; font-size: 1.05rem;">
            Enter live 2nd-innings chase conditions to compute calibrated winning probabilities 
            using the trained <code>pipe.pkl</code> pipeline.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

if not model_loaded:
    st.error(f"⚠️ Failed to load machine learning model pipeline: {model_err}")
    st.stop()

# Match Configuration Form
with st.container():
    st.markdown("### 🏟️ Match Setup & Current State")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        batting_team = st.selectbox(
            "🏏 Batting Team (Chasing)",
            options=SUPPORTED_TEAMS,
            index=0,
            help="Team currently batting in the 2nd innings."
        )
    with col2:
        # Default bowling team to a different team
        default_bowling_idx = 1 if SUPPORTED_TEAMS[0] == batting_team else 0
        bowling_team_options = [t for t in SUPPORTED_TEAMS if t != batting_team]
        bowling_team = st.selectbox(
            "🎯 Bowling Team (Defending)",
            options=bowling_team_options,
            index=0,
            help="Team defending their 1st innings total."
        )
    with col3:
        # Sort cities with common Indian venues first
        popular_cities = ['Mumbai', 'Chennai', 'Kolkata', 'Bengaluru', 'Delhi', 'Ahmedabad', 'Hyderabad', 'Jaipur', 'Lucknow', 'Mohali']
        other_cities = [c for c in SUPPORTED_CITIES if c not in popular_cities]
        city_options = popular_cities + other_cities
        city = st.selectbox(
            "📍 Host City",
            options=city_options,
            index=0,
            help="Host city of the fixture (encoded by model's OneHotEncoder)."
        )

    col4, col5, col6, col7 = st.columns(4)
    with col4:
        target = st.number_input(
            "🎯 Target Score",
            min_value=50,
            max_value=320,
            value=180,
            step=1,
            help="Target runs set by 1st innings batting team."
        )
    with col5:
        current_score = st.number_input(
            "📊 Current Score",
            min_value=0,
            max_value=320,
            value=115,
            step=1,
            help="Runs scored so far by the chasing team."
        )
    with col6:
        overs = st.number_input(
            "⏱️ Overs Completed",
            min_value=0.0,
            max_value=20.0,
            value=13.0,
            step=0.1,
            format="%.1f",
            help="Overs completed (e.g., 13.0, 13.1 ... 13.5)."
        )
    with col7:
        wickets_out = st.number_input(
            "🔴 Wickets Lost",
            min_value=0,
            max_value=10,
            value=3,
            step=1,
            help="Wickets lost by chasing team (0 to 10)."
        )

# Derive in-play parameters for live display
completed_overs = int(overs)
completed_balls = round((overs % 1) * 10)
balls_bowled = completed_overs * 6 + completed_balls
balls_left = max(0, 120 - balls_bowled)
runs_left = target - current_score
wickets_left = max(0, 10 - wickets_out)
crr = round((current_score * 6.0) / balls_bowled, 2) if balls_bowled > 0 else 0.0
rrr = round((runs_left * 6.0) / balls_left, 2) if balls_left > 0 else 0.0

# Live Telemetry Strip
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### ⚡ Live In-Play Derived Telemetry")
mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
with mcol1:
    render_kpi_card("Runs Needed", f"{runs_left}", f"Target: {target}", "Remaining")
with mcol2:
    render_kpi_card("Balls Left", f"{balls_left}", f"{overs} ov bowled", "Balls")
with mcol3:
    render_kpi_card("Wickets in Hand", f"{wickets_left}", f"{wickets_out} down", "Wickets")
with mcol4:
    render_kpi_card("Current RR (CRR)", f"{crr}", "Runs/Over", "Pace")
with mcol5:
    render_kpi_card("Required RR (RRR)", f"{rrr}", "Runs/Over", "Target Pace")

st.markdown("<br>", unsafe_allow_html=True)

# Prediction Button
predict_btn = st.button("⚡ PREDICT MATCH WINNER", use_container_width=True)

if predict_btn:
    result = predict_match_outcome(
        batting_team=batting_team,
        bowling_team=bowling_team,
        city=city,
        target=target,
        current_score=current_score,
        overs=overs,
        wickets_out=wickets_out
    )

    if not result["success"]:
        st.error(f"❌ Prediction Error: {result['error']}")
    else:
        bat_prob = result["batting_win_prob"]
        bowl_prob = result["bowling_win_prob"]
        winner = result["predicted_winner"]

        st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 1.5rem 0;'>", unsafe_allow_html=True)
        
        # Winner Banner
        winner_accent = "#8B5CF6" if winner == batting_team else "#FF6B00"
        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center; border-left: 6px solid {winner_accent}; padding: 1.8rem 1rem;">
                <span class="badge-pill badge-orange" style="font-size: 0.85rem;">Predicted Match Winner</span>
                <h1 style="margin: 0.5rem 0 0.2rem 0; font-size: 2.5rem; color: #FFFFFF !important;">
                    {winner} 🏆
                </h1>
                <p style="color: #94A3B8; font-size: 1.05rem; margin-bottom: 0;">
                    Model Confidence: <span style="color: {winner_accent}; font-weight: 800; font-size: 1.25rem;">
                    {max(bat_prob, bowl_prob)}%
                    </span>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Probability Split Columns
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.markdown(
                f"""
                <div class="prob-card prob-card-batting">
                    <span class="badge-pill badge-purple">Chasing Team</span>
                    <h3 style="margin: 0.5rem 0 0.2rem 0; color: #FFFFFF !important;">{batting_team}</h3>
                    <div class="prob-pct" style="color: #A78BFA;">{bat_prob}%</div>
                    <div style="color: #94A3B8; font-size: 0.85rem;">Win Probability</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with res_col2:
            st.markdown(
                f"""
                <div class="prob-card prob-card-bowling">
                    <span class="badge-pill badge-orange">Defending Team</span>
                    <h3 style="margin: 0.5rem 0 0.2rem 0; color: #FFFFFF !important;">{bowling_team}</h3>
                    <div class="prob-pct" style="color: #FF8533;">{bowl_prob}%</div>
                    <div style="color: #94A3B8; font-size: 0.85rem;">Win Probability</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Plotly Probability Donut Chart & Progression
        st.markdown("<br>", unsafe_allow_html=True)
        chart_col1, chart_col2 = st.columns([1, 1.2])

        with chart_col1:
            st.markdown("#### 🎯 Probability Distribution")
            donut_fig = go.Figure(data=[go.Pie(
                labels=[batting_team, bowling_team],
                values=[bat_prob, bowl_prob],
                hole=0.65,
                marker=dict(colors=['#8B5CF6', '#FF6B00'], line=dict(color='#0F172A', width=3)),
                textinfo='label+percent',
                textposition='outside',
                textfont=dict(size=13, color='#FFFFFF')
            )])
            donut_fig.update_layout(
                showlegend=False,
                height=320,
                annotations=[dict(text=f"<b>{max(bat_prob, bowl_prob)}%</b>", x=0.5, y=0.5, font_size=24, font_color='#FFFFFF', showarrow=False)]
            )
            apply_plotly_theme(donut_fig)
            st.plotly_chart(donut_fig, use_container_width=True)

        with chart_col2:
            st.markdown("#### 🔍 Sensitivity Analysis (Wicket Scenarios)")
            # Simulate what happens if wickets fall right now
            wickets_scenarios = []
            probs_scenarios = []
            
            for extra_w in range(0, min(5, wickets_left + 1)):
                sim_w = wickets_out + extra_w
                sim_res = predict_match_outcome(
                    batting_team=batting_team,
                    bowling_team=bowling_team,
                    city=city,
                    target=target,
                    current_score=current_score,
                    overs=overs,
                    wickets_out=sim_w
                )
                label = f"{sim_w} Wkts (+{extra_w})" if extra_w > 0 else f"{sim_w} Wkts (Now)"
                wickets_scenarios.append(label)
                probs_scenarios.append(sim_res["batting_win_prob"])

            scenario_fig = go.Figure()
            scenario_fig.add_trace(go.Bar(
                x=wickets_scenarios,
                y=probs_scenarios,
                text=[f"{p}%" for p in probs_scenarios],
                textposition='outside',
                marker_color=['#10B981' if i == 0 else '#8B5CF6' for i in range(len(wickets_scenarios))],
                marker_line_color='rgba(255,255,255,0.2)',
                marker_line_width=1
            ))
            scenario_fig.update_layout(
                height=320,
                xaxis_title="Wickets Scenario",
                yaxis_title=f"{batting_team} Win %",
                yaxis_range=[0, 105]
            )
            apply_plotly_theme(scenario_fig)
            st.plotly_chart(scenario_fig, use_container_width=True)

        # Pipeline Feature Vector Transparency
        with st.expander("🔬 View Pipeline Input Feature Vector (pipe.pkl)"):
            st.markdown(
                """
                The table below shows the exact 9-column feature vector generated from user inputs and 
                passed directly into the <code>pipe.pkl</code> pre-trained pipeline:
                """
            )
            if result.get("features_df") is not None:
                st.dataframe(result["features_df"], use_container_width=True)
            st.info("Features order: `['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets_left', 'runs_total_x', 'crr', 'rrr']`")
