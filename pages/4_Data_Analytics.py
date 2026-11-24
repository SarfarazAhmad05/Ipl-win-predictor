"""
pages/4_Data_Analytics.py
-------------------------
Data Analytics: League-wide exploratory data analysis with interactive Plotly visualisations.
Covers toss impacts, venue dynamics, batting first vs chasing trends, and victory margins.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.ui_helpers import load_custom_css, render_sidebar_brand, apply_plotly_theme, render_kpi_card
from src.data_loader import load_matches_data
from src.analytics import (
    get_team_win_statistics,
    get_toss_impact_analysis,
    get_batting_first_vs_chasing_overall,
    get_venue_analytics
)

st.set_page_config(page_title="Data Analytics | IPL Win Predictor", page_icon="📊", layout="wide")
load_custom_css()
render_sidebar_brand()

matches_df = load_matches_data()
team_stats = get_team_win_statistics(matches_df)
toss_analysis = get_toss_impact_analysis(matches_df)
bat_vs_chase = get_batting_first_vs_chasing_overall(matches_df)
venue_stats = get_venue_analytics(matches_df)

st.markdown(
    """
    <div style="padding: 1rem 0;">
        <span class="badge-pill badge-blue">Page 4 • League Intelligence</span>
        <h1 style="margin: 0.4rem 0;">IPL Macro Data Analytics & Visualizations</h1>
        <p style="color: #94A3B8; font-size: 1.05rem;">
            Interactive exploration of match dynamics, toss decisions, venue profiles, and historical winning patterns.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Tabs for structured viewing
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Team Success & Win %",
    "🪙 Toss Dynamics",
    "🏟️ Venues & Conditions",
    "⚡ Bat First vs Chasing"
])

# TAB 1: Team Success & Win %
with tab1:
    st.markdown("### 🏆 Franchise Standing & Win Efficiency")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 1. Most Successful Teams (All-Time Match Wins)")
        fig1 = px.bar(
            team_stats.head(10),
            x='Team',
            y='Wins',
            color='Wins',
            text='Wins',
            color_continuous_scale=[[0, '#00D2FF'], [1, '#8B5CF6']]
        )
        fig1.update_traces(textposition='outside')
        fig1.update_layout(coloraxis_showscale=False, height=420, xaxis_tickangle=-45)
        apply_plotly_theme(fig1)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("#### 2. Franchise Winning Percentage (%)")
        filtered_teams = team_stats[team_stats['Matches Played'] >= 25].sort_values(by='Win %', ascending=False)
        fig2 = px.bar(
            filtered_teams,
            x='Team',
            y='Win %',
            color='Win %',
            text='Win %',
            color_continuous_scale=[[0, '#FF6B00'], [1, '#10B981']]
        )
        fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig2.update_layout(coloraxis_showscale=False, height=420, xaxis_tickangle=-45)
        apply_plotly_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

# TAB 2: Toss Dynamics
with tab2:
    st.markdown("### 🪙 Toss Impact on Match Outcomes")
    tcol1, tcol2 = st.columns([1, 1.3])
    
    with tcol1:
        st.markdown("#### 4. Toss Winner vs Match Winner")
        toss_labels = ['Toss Winner Won Match', 'Toss Winner Lost Match']
        toss_values = [toss_analysis['toss_win_match_win_pct'], toss_analysis['toss_loss_match_win_pct']]
        
        fig_toss_pie = go.Figure(data=[go.Pie(
            labels=toss_labels,
            values=toss_values,
            hole=0.6,
            marker=dict(colors=['#10B981', '#64748B'], line=dict(color='#0F172A', width=3)),
            textinfo='label+percent'
        )])
        fig_toss_pie.update_layout(
            height=380,
            showlegend=False,
            annotations=[dict(text=f"<b>{toss_analysis['toss_win_match_win_pct']}%</b><br>Won Match", x=0.5, y=0.5, font_size=16, font_color='#FFFFFF', showarrow=False)]
        )
        apply_plotly_theme(fig_toss_pie)
        st.plotly_chart(fig_toss_pie, use_container_width=True)

    with tcol2:
        st.markdown("#### 5. Toss Decision Evolution Across Seasons")
        season_toss_df = toss_analysis['season_toss_df'].copy()
        season_toss_df['season'] = season_toss_df['season'].astype(int).astype(str)
        
        fig_toss_trend = go.Figure()
        if 'field' in season_toss_df.columns:
            fig_toss_trend.add_trace(go.Bar(
                x=season_toss_df['season'],
                y=season_toss_df['field'],
                name='Chose to Field',
                marker_color='#00D2FF'
            ))
        if 'bat' in season_toss_df.columns:
            fig_toss_trend.add_trace(go.Bar(
                x=season_toss_df['season'],
                y=season_toss_df['bat'],
                name='Chose to Bat',
                marker_color='#FF6B00'
            ))
        fig_toss_trend.update_layout(
            barmode='stack',
            height=380,
            xaxis_title="Season",
            yaxis_title="Decisions Count",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        apply_plotly_theme(fig_toss_trend)
        st.plotly_chart(fig_toss_trend, use_container_width=True)

    st.info(
        f"💡 **Toss Takeaway:** Teams choosing to **field** won **{toss_analysis['field_win_pct']}%** of their matches, "
        f"while teams choosing to **bat** won **{toss_analysis['bat_win_pct']}%**. Over recent years, captains overwhelmingly opt to field first."
    )

# TAB 3: Venues & Conditions
with tab3:
    st.markdown("### 🏟️ Venue-Wise Match Distribution & Chasing Success")
    st.markdown("#### 6. Top 15 Venues: Batting First vs Chasing Wins")
    
    top_venues = venue_stats.head(15).copy()
    fig_venues = go.Figure()
    fig_venues.add_trace(go.Bar(
        y=top_venues['Venue'],
        x=top_venues['Chasing Wins'],
        name='Chasing Wins',
        orientation='h',
        marker_color='#8B5CF6'
    ))
    fig_venues.add_trace(go.Bar(
        y=top_venues['Venue'],
        x=top_venues['Defended Wins'],
        name='Defended Wins',
        orientation='h',
        marker_color='#FF6B00'
    ))
    fig_venues.update_layout(
        barmode='stack',
        height=520,
        xaxis_title="Matches",
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    apply_plotly_theme(fig_venues)
    st.plotly_chart(fig_venues, use_container_width=True)

# TAB 4: Bat First vs Chasing
with tab4:
    st.markdown("### ⚡ Batting First vs Chasing (2nd Innings)")
    bcol1, bcol2 = st.columns([1, 1.4])
    
    with bcol1:
        st.markdown("#### 7. Overall League Split")
        fig_split_pie = go.Figure(data=[go.Pie(
            labels=['Chased (Bat 2nd)', 'Defended (Bat 1st)'],
            values=[bat_vs_chase['chasing_wins'], bat_vs_chase['bat_first_wins']],
            hole=0.6,
            marker=dict(colors=['#00D2FF', '#8B5CF6'], line=dict(color='#0F172A', width=3)),
            textinfo='label+percent'
        )])
        fig_split_pie.update_layout(
            height=380,
            showlegend=False,
            annotations=[dict(text=f"<b>{bat_vs_chase['total']:,}</b><br>Matches", x=0.5, y=0.5, font_size=18, font_color='#FFFFFF', showarrow=False)]
        )
        apply_plotly_theme(fig_split_pie)
        st.plotly_chart(fig_split_pie, use_container_width=True)

    with bcol2:
        st.markdown("#### 8. Historical Winning Trends by Season")
        season_split = bat_vs_chase['season_split_df'].copy()
        season_split['season'] = season_split['season'].astype(int).astype(str)
        
        fig_season_split = go.Figure()
        if 'Chased (Bat Second)' in season_split.columns:
            fig_season_split.add_trace(go.Bar(
                x=season_split['season'],
                y=season_split['Chased (Bat Second)'],
                name='Chasing Wins',
                marker_color='#00D2FF'
            ))
        if 'Defended (Bat First)' in season_split.columns:
            fig_season_split.add_trace(go.Bar(
                x=season_split['season'],
                y=season_split['Defended (Bat First)'],
                name='Defended Wins',
                marker_color='#8B5CF6'
            ))
        fig_season_split.update_layout(
            barmode='group',
            height=380,
            xaxis_title="Season",
            yaxis_title="Matches Won",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        apply_plotly_theme(fig_season_split)
        st.plotly_chart(fig_season_split, use_container_width=True)
