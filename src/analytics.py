"""
analytics.py
------------
Data analytics, aggregations, and business metrics for IPL matches and teams.
Powers interactive dashboard charts, KPI cards, and comparative statistics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


def get_overall_kpis(matches_df: pd.DataFrame, deliveries_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Computes top-level key performance indicators for the IPL dashboard.
    """
    total_matches = len(matches_df)
    seasons = sorted([int(s) for s in matches_df['season'].dropna().unique()])
    total_seasons = len(seasons)
    
    unique_teams = list(set(matches_df['team1'].dropna().unique()) | set(matches_df['team2'].dropna().unique()))
    total_teams = len(unique_teams)
    
    # Total runs & balls if deliveries provided
    total_runs = 0
    total_wickets = 0
    if deliveries_df is not None:
        total_runs = int(deliveries_df['runs_total'].sum())
        total_wickets = int(deliveries_df['wicket'].sum())

    # Most successful team by total wins
    wins_series = matches_df['winner'].value_counts()
    top_team = wins_series.index[0] if not wins_series.empty else "N/A"
    top_team_wins = int(wins_series.iloc[0]) if not wins_series.empty else 0

    return {
        "total_matches": total_matches,
        "total_seasons": total_seasons,
        "seasons_range": f"{min(seasons)} - {max(seasons)}" if seasons else "N/A",
        "total_teams": total_teams,
        "top_team": top_team,
        "top_team_wins": top_team_wins,
        "total_runs": total_runs,
        "total_wickets": total_wickets
    }


def get_team_win_statistics(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes matches played, wins, losses, ties/no-results, and winning percentage for all teams.
    """
    # Matches played
    played_team1 = matches_df['team1'].value_counts()
    played_team2 = matches_df['team2'].value_counts()
    matches_played = played_team1.add(played_team2, fill_value=0).astype(int)

    # Matches won
    wins = matches_df['winner'].value_counts()

    stats_df = pd.DataFrame({
        'Matches Played': matches_played,
        'Wins': wins
    }).fillna(0)
    
    stats_df['Wins'] = stats_df['Wins'].astype(int)
    stats_df['Losses'] = (stats_df['Matches Played'] - stats_df['Wins']).astype(int)
    stats_df['Win %'] = np.round((stats_df['Wins'] / stats_df['Matches Played']) * 100, 2)
    stats_df = stats_df.sort_values(by='Wins', ascending=False).reset_index()
    stats_df.rename(columns={'index': 'Team'}, inplace=True)
    return stats_df


def get_head_to_head_stats(matches_df: pd.DataFrame, team_a: str, team_b: str) -> Dict[str, Any]:
    """
    Computes head-to-head match statistics between two selected teams.
    """
    h2h_mask = (
        ((matches_df['team1'] == team_a) & (matches_df['team2'] == team_b)) |
        ((matches_df['team1'] == team_b) & (matches_df['team2'] == team_a))
    )
    h2h_matches = matches_df[h2h_mask].sort_values(by='date', ascending=False)
    
    total = len(h2h_matches)
    team_a_wins = len(h2h_matches[h2h_matches['winner'] == team_a])
    team_b_wins = len(h2h_matches[h2h_matches['winner'] == team_b])
    no_result = total - (team_a_wins + team_b_wins)

    recent_encounters = []
    for _, row in h2h_matches.head(5).iterrows():
        recent_encounters.append({
            "season": int(row['season']) if pd.notnull(row['season']) else "N/A",
            "date": str(row['date'])[:10] if pd.notnull(row['date']) else "N/A",
            "venue": row['venue'],
            "winner": row['winner'],
            "result_type": row['result_type'],
            "result_margin": row['result_margin'],
            "player_of_match": row['player_of_match']
        })

    return {
        "team_a": team_a,
        "team_b": team_b,
        "total_matches": total,
        "team_a_wins": team_a_wins,
        "team_b_wins": team_b_wins,
        "no_result": no_result,
        "team_a_win_pct": round((team_a_wins / total * 100), 1) if total > 0 else 0.0,
        "team_b_win_pct": round((team_b_wins / total * 100), 1) if total > 0 else 0.0,
        "recent_encounters": recent_encounters
    }


def get_team_detailed_profile(matches_df: pd.DataFrame, team: str) -> Dict[str, Any]:
    """
    Returns in-depth performance statistics for a specific franchise.
    """
    team_matches = matches_df[(matches_df['team1'] == team) | (matches_df['team2'] == team)].copy()
    total_played = len(team_matches)
    if total_played == 0:
        return {"total_played": 0}

    total_won = len(team_matches[team_matches['winner'] == team])
    total_lost = total_played - total_won
    win_rate = round((total_won / total_played) * 100, 2)

    # Batting First vs Chasing
    # In cricket, if team1 == team, they batted first; if team2 == team, they batted second (chased)
    batted_first_matches = team_matches[team_matches['team1'] == team]
    batted_first_won = len(batted_first_matches[batted_first_matches['winner'] == team])
    batted_first_total = len(batted_first_matches)
    bat_first_win_pct = round((batted_first_won / batted_first_total * 100), 1) if batted_first_total > 0 else 0.0

    chased_matches = team_matches[team_matches['team2'] == team]
    chased_won = len(chased_matches[chased_matches['winner'] == team])
    chased_total = len(chased_matches)
    chase_win_pct = round((chased_won / chased_total * 100), 1) if chased_total > 0 else 0.0

    # Season-wise performance
    seasons = sorted(team_matches['season'].dropna().unique())
    season_stats = []
    for s in seasons:
        s_matches = team_matches[team_matches['season'] == s]
        s_played = len(s_matches)
        s_won = len(s_matches[s_matches['winner'] == team])
        season_stats.append({
            "Season": str(int(s)),
            "Played": s_played,
            "Won": s_won,
            "Lost": s_played - s_won,
            "Win %": round((s_won / s_played) * 100, 1) if s_played > 0 else 0.0
        })

    # Top venues for this team
    venue_stats = []
    venues = team_matches['venue'].value_counts().head(8).index
    for v in venues:
        v_matches = team_matches[team_matches['venue'] == v]
        v_played = len(v_matches)
        v_won = len(v_matches[v_matches['winner'] == team])
        venue_stats.append({
            "Venue": v,
            "Played": v_played,
            "Won": v_won,
            "Win %": round((v_won / v_played) * 100, 1) if v_played > 0 else 0.0
        })

    return {
        "team": team,
        "total_played": total_played,
        "total_won": total_won,
        "total_lost": total_lost,
        "win_rate": win_rate,
        "batted_first_total": batted_first_total,
        "batted_first_won": batted_first_won,
        "bat_first_win_pct": bat_first_win_pct,
        "chased_total": chased_total,
        "chased_won": chased_won,
        "chase_win_pct": chase_win_pct,
        "season_stats": pd.DataFrame(season_stats),
        "venue_stats": pd.DataFrame(venue_stats)
    }


def get_toss_impact_analysis(matches_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes the correlation between toss winners and match winners.
    """
    total_matches = len(matches_df.dropna(subset=['winner', 'toss_winner']))
    toss_and_match_win = len(matches_df[matches_df['toss_winner'] == matches_df['winner']])
    toss_win_match_win_pct = round((toss_and_match_win / total_matches) * 100, 2) if total_matches > 0 else 0.0

    decision_counts = matches_df['toss_decision'].value_counts().to_dict()

    # Success rate when choosing to bat vs field
    field_chosen = matches_df[matches_df['toss_decision'] == 'field']
    field_won = len(field_chosen[field_chosen['toss_winner'] == field_chosen['winner']])
    field_win_pct = round((field_won / len(field_chosen)) * 100, 2) if len(field_chosen) > 0 else 0.0

    bat_chosen = matches_df[matches_df['toss_decision'] == 'bat']
    bat_won = len(bat_chosen[bat_chosen['toss_winner'] == bat_chosen['winner']])
    bat_win_pct = round((bat_won / len(bat_chosen)) * 100, 2) if len(bat_chosen) > 0 else 0.0

    # Season-wise toss decisions
    season_toss = matches_df.groupby(['season', 'toss_decision']).size().unstack(fill_value=0).reset_index()

    return {
        "total_matches": total_matches,
        "toss_and_match_win": toss_and_match_win,
        "toss_win_match_win_pct": toss_win_match_win_pct,
        "toss_loss_match_win_pct": round(100 - toss_win_match_win_pct, 2),
        "decision_counts": decision_counts,
        "field_win_pct": field_win_pct,
        "bat_win_pct": bat_win_pct,
        "season_toss_df": season_toss
    }


def get_batting_first_vs_chasing_overall(matches_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes overall league success of teams batting first vs teams chasing (batting second).
    In matches_df:
    result_type == 'runs' means team1 (batting first) won.
    result_type == 'wickets' means team2 (chasing) won.
    """
    valid = matches_df[matches_df['result_type'].isin(['runs', 'wickets'])].copy()
    total = len(valid)
    bat_first_wins = len(valid[valid['result_type'] == 'runs'])
    chasing_wins = len(valid[valid['result_type'] == 'wickets'])

    # By Season
    season_split = valid.groupby(['season', 'result_type']).size().unstack(fill_value=0).reset_index()
    if 'runs' in season_split.columns:
        season_split.rename(columns={'runs': 'Defended (Bat First)'}, inplace=True)
    if 'wickets' in season_split.columns:
        season_split.rename(columns={'wickets': 'Chased (Bat Second)'}, inplace=True)

    return {
        "total": total,
        "bat_first_wins": bat_first_wins,
        "chasing_wins": chasing_wins,
        "bat_first_pct": round((bat_first_wins / total) * 100, 2) if total > 0 else 0.0,
        "chasing_pct": round((chasing_wins / total) * 100, 2) if total > 0 else 0.0,
        "season_split_df": season_split
    }


def get_venue_analytics(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates match counts, bat first vs chasing wins, and top venues.
    """
    venue_counts = matches_df['venue'].value_counts().reset_index()
    venue_counts.columns = ['Venue', 'Total Matches']

    # Bat first vs chasing per venue
    runs_wins = matches_df[matches_df['result_type'] == 'runs']['venue'].value_counts()
    wickets_wins = matches_df[matches_df['result_type'] == 'wickets']['venue'].value_counts()

    venue_counts['Defended Wins'] = venue_counts['Venue'].map(runs_wins).fillna(0).astype(int)
    venue_counts['Chasing Wins'] = venue_counts['Venue'].map(wickets_wins).fillna(0).astype(int)
    venue_counts['Chasing Win %'] = np.round((venue_counts['Chasing Wins'] / venue_counts['Total Matches']) * 100, 1)

    return venue_counts
