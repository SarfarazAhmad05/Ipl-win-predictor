"""
pages/3_Team_Analytics.py
-------------------------
Team Analytics: In-depth franchise profiles, historical winning percentages,
season-by-season performance, batting first vs chasing success, and head-to-head records.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.ui_helpers import load_custom_css, render_sidebar_brand, apply_plotly_theme, render_kpi_card
from src.data_loader import load_matches_data, get_active_teams
from src.analytics import get_team_detailed_profile, get_head_to_head_stats

st.set_page_config(page_title="Team Analytics | IPL Win Predictor", page_icon="🏆", layout="wide")
load_custom_css()
render_sidebar_brand()

matches_df = load_matches_data()
all_teams = get_active_teams(matches_df)

st.markdown(
    """
    <div style="padding: 1rem 0;">
        <span class="badge-pill badge-purple">Page 3 • Franchise Telemetry</span>
        <h1 style="margin: 0.4rem 0;">IPL Franchise Analytics & Head-to-Head</h1>
        <p style="color: #94A3B8; font-size: 1.05rem;">
            Deep-dive into team historical performance, defending vs chasing records, venue advantages, and rivalries.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Team Selection
sel_col1, sel_col2 = st.columns([1.5, 1])
with sel_col1:
    selected_team = st.selectbox(
        "🏆 Select Franchise Profile",
        options=all_teams,
        index=all_teams.index("Chennai Super Kings") if "Chennai Super Kings" in all_teams else 0
    )

profile = get_team_detailed_profile(matches_df, selected_team)

if profile["total_played"] == 0:
    st.warning("No matches recorded for this team.")
    st.stop()

# Profile KPI Header
st.markdown(f"### 📊 Overall Record: <span style='color: #00D2FF;'>{selected_team}</span>", unsafe_allow_html=True)
pcol1, pcol2, pcol3, pcol4 = st.columns(4)
with pcol1:
    render_kpi_card("Matches Played", f"{profile['total_played']}", "All Seasons", "Total")
with pcol2:
    render_kpi_card("Matches Won", f"{profile['total_won']}", f"{profile['total_lost']} Losses", "Wins")
with pcol3:
    render_kpi_card("Win Percentage", f"{profile['win_rate']}%", "Win Efficiency", "Rate")
with pcol4:
    render_kpi_card("Defend vs Chase", f"{profile['batted_first_won']}W / {profile['chased_won']}W", f"Bat 1st: {profile['bat_first_win_pct']}% | Chase: {profile['chase_win_pct']}%", "Split")

st.markdown("<br>", unsafe_allow_html=True)

# Season-wise performance and Bat 1st vs Chase Split
row2_col1, row2_col2 = st.columns([1.3, 1])

with row2_col1:
    st.markdown("#### 📅 Season-by-Season Performance")
    season_df = profile["season_stats"]
    
    if not season_df.empty:
        fig_season = go.Figure()
        fig_season.add_trace(go.Bar(
            x=season_df['Season'],
            y=season_df['Won'],
            name='Wins',
            marker_color='#10B981',
            marker_line_color='rgba(255,255,255,0.2)',
            marker_line_width=1
        ))
        fig_season.add_trace(go.Bar(
            x=season_df['Season'],
            y=season_df['Lost'],
            name='Losses',
            marker_color='#EF4444',
            marker_line_color='rgba(255,255,255,0.2)',
            marker_line_width=1
        ))
        fig_season.add_trace(go.Scatter(
            x=season_df['Season'],
            y=season_df['Win %'],
            name='Win %',
            yaxis='y2',
            mode='lines+markers',
            line=dict(color='#00D2FF', width=3),
            marker=dict(size=7, color='#FFFFFF')
        ))
        fig_season.update_layout(
            barmode='stack',
            height=380,
            yaxis=dict(title="Matches"),
            yaxis2=dict(title="Win %", overlaying='y', side='right', range=[0, 100], showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        apply_plotly_theme(fig_season)
        st.plotly_chart(fig_season, use_container_width=True)

with row2_col2:
    st.markdown("#### ⚖️ Batting First vs Chasing Success")
    split_labels = ['Defended (Bat 1st)', 'Chased (Bat 2nd)']
    split_values = [profile['batted_first_won'], profile['chased_won']]
    
    fig_split = go.Figure(data=[go.Pie(
        labels=split_labels,
        values=split_values,
        hole=0.6,
        marker=dict(colors=['#8B5CF6', '#FF6B00'], line=dict(color='#0F172A', width=3)),
        textinfo='label+percent',
        textposition='outside'
    )])
    fig_split.update_layout(
        height=380,
        showlegend=False,
        annotations=[dict(text=f"<b>{profile['total_won']} Wins</b>", x=0.5, y=0.5, font_size=18, font_color='#FFFFFF', showarrow=False)]
    )
    apply_plotly_theme(fig_split)
    st.plotly_chart(fig_split, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Venue Performance for this team
st.markdown("#### 🏟️ Top Venues Performance")
venue_df = profile["venue_stats"]
if not venue_df.empty:
    fig_venue = px.bar(
        venue_df,
        x='Venue',
        y='Played',
        color='Win %',
        text='Win %',
        color_continuous_scale=[[0, '#1E293B'], [0.5, '#8B5CF6'], [1, '#00D2FF']],
    )
    fig_venue.update_traces(
        texttemplate='%{text:.1f}% Win',
        textposition='outside',
        marker_line_color='rgba(255,255,255,0.15)',
        marker_line_width=1
    )
    fig_venue.update_layout(height=380, xaxis_tickangle=-35)
    apply_plotly_theme(fig_venue)
    st.plotly_chart(fig_venue, use_container_width=True)

st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 2rem 0;'>", unsafe_allow_html=True)

# Head-to-Head Section
st.markdown("### ⚔️ Head-to-Head Franchise Comparison")
h2h_opponents = [t for t in all_teams if t != selected_team]

hcol1, hcol2 = st.columns([1, 2])
with hcol1:
    opponent = st.selectbox(
        "Select Opponent Franchise",
        options=h2h_opponents,
        index=h2h_opponents.index("Mumbai Indians") if "Mumbai Indians" in h2h_opponents else 0
    )
    
    h2h = get_head_to_head_stats(matches_df, selected_team, opponent)
    
    st.markdown(
        f"""
        <div class="glass-card" style="margin-top: 1rem;">
            <div style="font-size: 0.85rem; text-transform: uppercase; color: #94A3B8; font-weight: 700;">H2H Summary</div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #FFFFFF; margin: 0.4rem 0;">
                {h2h['team_a_wins']} - {h2h['team_b_wins']}
            </div>
            <div style="color: #CBD5E1; font-size: 0.9rem;">
                <b>{selected_team}:</b> {h2h['team_a_win_pct']}% ({h2h['team_a_wins']} wins)<br>
                <b>{opponent}:</b> {h2h['team_b_win_pct']}% ({h2h['team_b_wins']} wins)<br>
                <b>Total Encounters:</b> {h2h['total_matches']} matches
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with hcol2:
    if h2h['total_matches'] > 0:
        fig_h2h = go.Figure(data=[go.Bar(
            y=[selected_team, opponent],
            x=[h2h['team_a_wins'], h2h['team_b_wins']],
            orientation='h',
            text=[f"{h2h['team_a_wins']} Wins ({h2h['team_a_win_pct']}%)", f"{h2h['team_b_wins']} Wins ({h2h['team_b_win_pct']}%)"],
            textposition='outside',
            marker_color=['#8B5CF6', '#FF6B00'],
            marker_line_color='rgba(255,255,255,0.2)',
            marker_line_width=1
        )])
        fig_h2h.update_layout(
            height=260,
            xaxis_title="Total Matches Won",
            xaxis_range=[0, max(h2h['team_a_wins'], h2h['team_b_wins']) * 1.3]
        )
        apply_plotly_theme(fig_h2h)
        st.plotly_chart(fig_h2h, use_container_width=True)

        if h2h["recent_encounters"]:
            st.markdown("##### Recent Encounters")
            recent_df = pd.DataFrame(h2h["recent_encounters"])
            st.dataframe(recent_df, use_container_width=True, height=180)
    else:
        st.info("No recorded historical matches found between these two teams.")
