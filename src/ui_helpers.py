"""
ui_helpers.py
-------------
Reusable Streamlit UI utilities, styling injectors, and Plotly theme configurators.
"""

import os
from typing import Any
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Brand color constants
IPL_PURPLE = "#8B5CF6"
IPL_BLUE = "#00D2FF"
IPL_ORANGE = "#FF6B00"
IPL_GOLD = "#F59E0B"
IPL_GREEN = "#10B981"
IPL_DARK_BG = "#0B1120"
IPL_CARD_BG = "#131C31"

COLOR_PALETTE = [IPL_BLUE, IPL_PURPLE, IPL_ORANGE, IPL_GOLD, "#EC4899", "#10B981", "#3B82F6", "#F97316"]


def load_custom_css():
    """
    Injects custom stylesheet into the Streamlit app.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    css_path = os.path.join(base_dir, "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_sidebar_brand():
    """
    Renders top branding and navigation info in the sidebar.
    """
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; padding: 1.2rem 0 1rem 0;">
                <div style="display: inline-block; background: linear-gradient(135deg, #FF6B00, #8B5CF6); border-radius: 50%; padding: 3px; box-shadow: 0 0 20px rgba(139, 92, 246, 0.4);">
                    <div style="background: #0B1120; border-radius: 50%; width: 68px; height: 68px; display: flex; align-items: center; justify-content: center; font-size: 30px;">
                        🏏
                    </div>
                </div>
                <h2 style="margin: 0.8rem 0 0.2rem 0; font-size: 1.4rem; letter-spacing: -0.02em;">
                    IPL <span style="color: #00D2FF;">PREDICTOR</span>
                </h2>
                <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;">
                    Machine Learning Engine
                </div>
                <div style="margin-top: 0.6rem;">
                    <span class="badge-pill badge-purple">v2.4 Production</span>
                    <span class="badge-pill badge-blue">1,243 Matches</span>
                </div>
            </div>
            <hr style="border-color: rgba(255,255,255,0.08); margin: 0.8rem 0 1.2rem 0;">
            """,
            unsafe_allow_html=True
        )


def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    """
    Applies consistent dark sports-analytics theme to any Plotly figure.
    """
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#F8FAFC", size=12),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(255, 255, 255, 0.1)",
            tickfont=dict(color="#94A3B8")
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(255, 255, 255, 0.1)",
            tickfont=dict(color="#94A3B8")
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.6)",
            bordercolor="rgba(255, 255, 255, 0.08)",
            font=dict(color="#F8FAFC")
        ),
        hoverlabel=dict(
            bgcolor="#0F172A",
            font_size=12,
            font_family="Plus Jakarta Sans"
        )
    )
    return fig


def render_kpi_card(label: str, value, subtext: str = "", badge: str = ""):
    """
    Renders an HTML glassmorphism KPI card.
    """
    badge_html = f'<span class="badge-pill badge-blue" style="margin-left: auto;">{badge}</span>' if badge else ""
    html = f"""
    <div class="kpi-container">
        <div style="display: flex; align-items: center;">
            <span class="kpi-label">{label}</span>
            {badge_html}
        </div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-subtext">{subtext}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
