"""
data_loader.py
--------------
Efficient data loading and caching utility for IPL matches and deliveries datasets.
"""

import os
import pandas as pd
from typing import Tuple, Optional

try:
    import streamlit as st
    cache_decorator = st.cache_data
except ImportError:
    # Fallback if streamlit is not in scope
    def cache_decorator(func):
        return func


@cache_decorator
def load_matches_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Loads and cleans matches.csv.
    """
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "data", "matches.csv")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Matches dataset not found at: {filepath}")

    df = pd.read_csv(filepath)
    # Ensure correct data types
    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # Fill missing cities with 'Unknown'
    df['city'] = df['city'].fillna('Unknown')
    return df


@cache_decorator
def load_deliveries_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Loads deliveries.csv.
    """
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "data", "deliveries.csv")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Deliveries dataset not found at: {filepath}")

    df = pd.read_csv(filepath)
    return df


@cache_decorator
def get_all_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience method to retrieve both matches and deliveries dataframes.
    """
    matches = load_matches_data()
    deliveries = load_deliveries_data()
    return matches, deliveries


def get_active_teams(matches_df: pd.DataFrame) -> list:
    """
    Returns unique active IPL teams.
    """
    teams = sorted(list(set(matches_df['team1'].dropna().unique()) | set(matches_df['team2'].dropna().unique())))
    return teams


def get_all_venues(matches_df: pd.DataFrame) -> list:
    """
    Returns unique IPL venues.
    """
    return sorted([v for v in matches_df['venue'].dropna().unique() if v])


def get_all_cities(matches_df: pd.DataFrame) -> list:
    """
    Returns unique cities where IPL matches have taken place.
    """
    cities = sorted([c for c in matches_df['city'].dropna().unique() if c and c != 'Unknown'])
    return cities
