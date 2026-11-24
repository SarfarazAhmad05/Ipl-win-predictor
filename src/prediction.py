"""
prediction.py
-------------
Reusable prediction module for IPL Win Predictor.
Loads and caches the trained scikit-learn pipeline (pipe.pkl)
and generates validated win probabilities for chasing matches.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional

# Supported teams trained in the pipeline
SUPPORTED_TEAMS: List[str] = [
    'Chennai Super Kings',
    'Delhi Capitals',
    'Gujarat Titans',
    'Kolkata Knight Riders',
    'Lucknow Super Giants',
    'Mumbai Indians',
    'Punjab Kings',
    'Rajasthan Royals',
    'Royal Challengers Bengaluru',
    'Sunrisers Hyderabad'
]

# Supported cities extracted from the pipeline's OneHotEncoder categories
SUPPORTED_CITIES: List[str] = [
    'Abu Dhabi', 'Ahmedabad', 'Bangalore', 'Bengaluru', 'Bloemfontein',
    'Cape Town', 'Centurion', 'Chandigarh', 'Chennai', 'Cuttack',
    'Delhi', 'Dharamsala', 'Dubai', 'Durban', 'East London',
    'Guwahati', 'Hyderabad', 'Indore', 'Jaipur', 'Johannesburg',
    'Kimberley', 'Kolkata', 'Lucknow', 'Mohali', 'Mumbai',
    'Navi Mumbai', 'New Chandigarh', 'Port Elizabeth', 'Pune', 'Raipur',
    'Ranchi', 'Sharjah', 'Visakhapatnam', 'Unknown'
]

# Global cache for the loaded pipeline singleton
_PIPELINE = None


def get_pipeline(model_path: Optional[str] = None):
    """
    Loads the trained ML pipeline (pipe.pkl) once and caches it.
    """
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    if model_path is None:
        # Search relative to this file or root project directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "pipe.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    with open(model_path, "rb") as f:
        _PIPELINE = pickle.load(f)

    return _PIPELINE


def validate_inputs(
    batting_team: str,
    bowling_team: str,
    city: str,
    target: float,
    current_score: float,
    overs: float,
    wickets_out: int
) -> Tuple[bool, Optional[str]]:
    """
    Validates user inputs before feeding into the ML model.
    """
    if batting_team == bowling_team:
        return False, "Batting team and Bowling team cannot be the same."

    if batting_team not in SUPPORTED_TEAMS:
        return False, f"Batting team '{batting_team}' is not recognized."

    if bowling_team not in SUPPORTED_TEAMS:
        return False, f"Bowling team '{bowling_team}' is not recognized."

    if target <= 0:
        return False, "Target score must be a positive integer."

    if current_score < 0:
        return False, "Current score cannot be negative."

    if overs < 0 or overs > 20:
        return False, "Overs completed must be between 0 and 20."

    # Validate fractional over: decimal part can only be .0 through .5 (balls 0 to 5)
    fractional = round((overs % 1) * 10)
    if fractional > 5:
        return False, f"Invalid cricket over '{overs}'. An over consists of 6 balls (e.g. 5.0 to 5.5)."

    if wickets_out < 0 or wickets_out > 10:
        return False, "Wickets lost must be between 0 and 10."

    return True, None


def predict_match_outcome(
    batting_team: str,
    bowling_team: str,
    city: str,
    target: float,
    current_score: float,
    overs: float,
    wickets_out: int,
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the trained pipeline on live match chase inputs.
    
    Returns a dictionary containing:
      - success (bool)
      - error (str or None)
      - batting_win_prob (float)
      - bowling_win_prob (float)
      - predicted_winner (str)
      - input_summary (dict)
      - features_df (DataFrame)
    """
    is_valid, err_msg = validate_inputs(
        batting_team, bowling_team, city, target, current_score, overs, wickets_out
    )
    if not is_valid:
        return {
            "success": False,
            "error": err_msg,
            "batting_win_prob": 0.0,
            "bowling_win_prob": 0.0,
            "predicted_winner": None,
            "input_summary": {},
            "features_df": None
        }

    # Derive match features
    completed_overs = int(overs)
    completed_balls = round((overs % 1) * 10)
    balls_bowled = completed_overs * 6 + completed_balls
    balls_left = max(0, 120 - balls_bowled)
    runs_left = target - current_score
    wickets_left = max(0, 10 - wickets_out)
    runs_total_x = target

    # Handle boundary cricket conditions
    if runs_left <= 0:
        # Target achieved - Batting team wins
        return {
            "success": True,
            "error": None,
            "batting_win_prob": 100.0,
            "bowling_win_prob": 0.0,
            "predicted_winner": batting_team,
            "input_summary": {
                "runs_left": runs_left,
                "balls_left": balls_left,
                "wickets_left": wickets_left,
                "crr": round((current_score * 6) / max(1, balls_bowled), 2),
                "rrr": 0.0,
                "note": "Target reached! Batting team has won."
            },
            "features_df": None
        }

    if wickets_left == 0:
        # All out - Bowling team wins
        return {
            "success": True,
            "error": None,
            "batting_win_prob": 0.0,
            "bowling_win_prob": 100.0,
            "predicted_winner": bowling_team,
            "input_summary": {
                "runs_left": runs_left,
                "balls_left": balls_left,
                "wickets_left": 0,
                "crr": round((current_score * 6) / max(1, balls_bowled), 2),
                "rrr": 0.0,
                "note": "All out! Bowling team has won."
            },
            "features_df": None
        }

    if balls_left == 0 and runs_left > 0:
        # Out of balls - Bowling team wins
        return {
            "success": True,
            "error": None,
            "batting_win_prob": 0.0,
            "bowling_win_prob": 100.0,
            "predicted_winner": bowling_team,
            "input_summary": {
                "runs_left": runs_left,
                "balls_left": 0,
                "wickets_left": wickets_left,
                "crr": round((current_score * 6) / 120, 2),
                "rrr": 0.0,
                "note": "Innings complete! Bowling team defended successfully."
            },
            "features_df": None
        }

    # Rates calculation
    crr = (current_score * 6.0) / balls_bowled if balls_bowled > 0 else 0.0
    rrr = (runs_left * 6.0) / balls_left if balls_left > 0 else 0.0

    # Format input DataFrame matching the exact order and columns of pipe.pkl
    # ['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets_left', 'runs_total_x', 'crr', 'rrr']
    features = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city if city in SUPPORTED_CITIES else 'Unknown'],
        'runs_left': [float(runs_left)],
        'balls_left': [float(balls_left)],
        'wickets_left': [float(wickets_left)],
        'runs_total_x': [float(runs_total_x)],
        'crr': [float(crr)],
        'rrr': [float(rrr)]
    })

    try:
        pipe = get_pipeline(model_path)
        # Predict probabilities
        probabilities = pipe.predict_proba(features)[0]
        # Class 0: Bowling Team Wins, Class 1: Batting Team Wins
        bowling_prob = round(float(probabilities[0]) * 100, 1)
        batting_prob = round(float(probabilities[1]) * 100, 1)

        predicted_winner = batting_team if batting_prob >= bowling_prob else bowling_team

        return {
            "success": True,
            "error": None,
            "batting_win_prob": batting_prob,
            "bowling_win_prob": bowling_prob,
            "predicted_winner": predicted_winner,
            "input_summary": {
                "runs_left": int(runs_left),
                "balls_left": int(balls_left),
                "wickets_left": int(wickets_left),
                "target": int(target),
                "current_score": int(current_score),
                "crr": round(crr, 2),
                "rrr": round(rrr, 2),
            },
            "features_df": features
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Model inference error: {str(e)}",
            "batting_win_prob": 0.0,
            "bowling_win_prob": 0.0,
            "predicted_winner": None,
            "input_summary": {},
            "features_df": features
        }
