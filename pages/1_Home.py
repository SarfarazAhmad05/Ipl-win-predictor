"""
pages/1_Home.py
---------------
Home Dashboard: Overview of IPL historical datasets, league statistics,
franchise winning percentages, and historical trends.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.ui_helpers import load_custom_css, render_sidebar_brand, apply_plotly_theme, render_kpi_card, COLOR_PALETTE
from src.data_loader import load_matches_data, load_deliveries_data
from src.analytics import get_overall_kpis, get_team_win_statistics

st.set_page_config(page_title="Home Dashboard | IPL Win Predictor", page_icon="🏏", layout="wide")
load_custom_css()
render_sidebar_brand()

matches_df = load_matches_data()
deliveries_df = load_deliveries_data()
kpis = get_overall_kpis(matches_df, deliveries_df)
team_stats = get_team_win_statistics(matches_df)

# Page Title
st.markdown(
    """
    <div style="padding: 1rem 0;">
        <span class="badge-pill badge-blue">Page 1 • Home Dashboard</span>
        <h1 style="margin: 0.4rem 0;">IPL Historical Overview & League Records</h1>
        <p style="color: #94A3B8; font-size: 1.05rem;">
            Comprehensive historical telemetry and franchise records computed directly from the 
            official <code>matches.csv</code> and <code>deliveries.csv</code> datasets.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# KPI Metrics Grid
kcol1, kcol2, kcol3, kcol4, kcol5 = st.columns(5)
with kcol1:
    render_kpi_card("Total Matches", f"{kpis['total_matches']:,}", "All-Time Fixtures", "Official")
with kcol2:
    render_kpi_card("Total Seasons", f"{kpis['total_seasons']}", kpis['seasons_range'], "Editions")
with kcol3:
    render_kpi_card("Unique Franchises", f"{kpis['total_teams']}", "Historic + Active", "Teams")
with kcol4:
    render_kpi_card("Most Match Wins", f"{kpis['top_team_wins']}", kpis['top_team'], "Leader")
with kcol5:
    render_kpi_card("Wickets Taken", f"{kpis['total_wickets']:,}", "All Deliveries", "Dismissals")

st.markdown("<br>", unsafe_allow_html=True)

# Top 2 Visualizations: Most Successful Teams and Win %
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.markdown("#### 🏆 Most Successful IPL Franchises (All-Time Wins)")
    top10_teams = team_stats.head(10).sort_values(by='Wins', ascending=True)
    
    fig_wins = px.bar(
        top10_teams,
        x='Wins',
        y='Team',
        orientation='h',
        text='Wins',
        color='Wins',
        color_continuous_scale=[[0, '#00D2FF'], [0.5, '#8B5CF6'], [1, '#FF6B00']],
    )
    fig_wins.update_traces(
        textposition='outside',
        marker_line_color='rgba(255,255,255,0.15)',
        marker_line_width=1
    )
    fig_wins.update_layout(coloraxis_showscale=False, height=420)
    apply_plotly_theme(fig_wins)
    st.plotly_chart(fig_wins, use_container_width=True)

with col_right:
    st.markdown("#### 📈 Franchise Winning Percentages (%)")
    # Filter teams with at least 20 matches for fair comparison
    qualifying_teams = team_stats[team_stats['Matches Played'] >= 20].sort_values(by='Win %', ascending=False)
    
    fig_pct = px.bar(
        qualifying_teams,
        x='Team',
        y='Win %',
        text='Win %',
        color='Win %',
        color_continuous_scale=[[0, '#1E293B'], [0.5, '#8B5CF6'], [1, '#10B981']]
    )
    fig_pct.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside',
        marker_line_color='rgba(255,255,255,0.15)',
        marker_line_width=1
    )
    fig_pct.update_layout(coloraxis_showscale=False, height=420, xaxis_tickangle=-45)
    apply_plotly_theme(fig_pct)
    st.plotly_chart(fig_pct, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Season-wise matches and total runs
st.markdown("#### 📅 Historical Season Progression (2008 – Present)")
season_matches = matches_df.groupby('season').size().reset_index(name='Matches Count')
season_matches['season'] = season_matches['season'].astype(int).astype(str)

fig_season = go.Figure()
fig_season.add_trace(go.Bar(
    x=season_matches['season'],
    y=season_matches['Matches Count'],
    name='Matches Played',
    marker_color='#8B5CF6',
    marker_line_color='rgba(255,255,255,0.2)',
    marker_line_width=1
))
fig_season.update_layout(
    height=340,
    xaxis_title="IPL Season",
    yaxis_title="Total Matches Played",
    bargap=0.3
)
apply_plotly_theme(fig_season)
st.plotly_chart(fig_season, use_container_width=True)

# Full Team Performance Table
st.markdown("#### 📋 Complete Franchise Performance Table")
st.dataframe(
    team_stats.style.background_gradient(subset=['Wins', 'Win %'], cmap='Blues'),
    use_container_width=True,
    height=360
)
