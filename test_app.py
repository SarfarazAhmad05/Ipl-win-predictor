"""
test_app.py
-----------
Comprehensive automated test suite for IPL Win Predictor.
Validates data loaders, pipeline inference, edge cases, analytics, and page scripts.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.data_loader import load_matches_data, load_deliveries_data, get_active_teams, get_all_cities
from src.prediction import (
    get_pipeline,
    predict_match_outcome,
    validate_inputs,
    SUPPORTED_TEAMS,
    SUPPORTED_CITIES
)
from src.analytics import (
    get_overall_kpis,
    get_team_win_statistics,
    get_head_to_head_stats,
    get_team_detailed_profile,
    get_toss_impact_analysis,
    get_batting_first_vs_chasing_overall,
    get_venue_analytics
)


class TestIPLWinPredictor(unittest.TestCase):

    def test_01_data_loading(self):
        """Verify matches and deliveries datasets load properly."""
        matches = load_matches_data()
        self.assertIsInstance(matches, pd.DataFrame)
        self.assertEqual(len(matches), 1243)
        self.assertIn('match_id', matches.columns)
        self.assertIn('winner', matches.columns)

        deliveries = load_deliveries_data()
        self.assertIsInstance(deliveries, pd.DataFrame)
        self.assertGreater(len(deliveries), 290000)
        self.assertIn('runs_total', deliveries.columns)
        print("✓ Test 01 Passed: Datasets loaded successfully.")

    def test_02_pipeline_loading(self):
        """Verify pipe.pkl loads and possesses expected structure."""
        pipe = get_pipeline()
        self.assertIsNotNone(pipe)
        self.assertTrue(hasattr(pipe, 'predict_proba'))
        self.assertTrue('step1' in pipe.named_steps)
        self.assertTrue('step2' in pipe.named_steps)
        print("✓ Test 02 Passed: pipe.pkl loaded with predict_proba supported.")

    def test_03_prediction_valid_chase(self):
        """Verify standard in-play chase prediction."""
        res = predict_match_outcome(
            batting_team='Royal Challengers Bengaluru',
            bowling_team='Chennai Super Kings',
            city='Bengaluru',
            target=180,
            current_score=110,
            overs=13.0,
            wickets_out=3
        )
        self.assertTrue(res['success'])
        self.assertIsNone(res['error'])
        self.assertGreater(res['batting_win_prob'], 0)
        self.assertGreater(res['bowling_win_prob'], 0)
        self.assertAlmostEqual(res['batting_win_prob'] + res['bowling_win_prob'], 100.0, places=1)
        self.assertIn(res['predicted_winner'], ['Royal Challengers Bengaluru', 'Chennai Super Kings'])
        print(f"✓ Test 03 Passed: Valid chase predicted {res['predicted_winner']} (RCB: {res['batting_win_prob']}%, CSK: {res['bowling_win_prob']}%).")

    def test_04_prediction_edge_cases(self):
        """Verify cricket edge cases (target achieved, all out, same team)."""
        # Edge Case 1: Same team
        res_same = predict_match_outcome('Mumbai Indians', 'Mumbai Indians', 'Mumbai', 180, 100, 10.0, 2)
        self.assertFalse(res_same['success'])
        self.assertIn("cannot be the same", res_same['error'])

        # Edge Case 2: Target already reached
        res_won = predict_match_outcome('Kolkata Knight Riders', 'Delhi Capitals', 'Kolkata', 150, 151, 16.4, 4)
        self.assertTrue(res_won['success'])
        self.assertEqual(res_won['batting_win_prob'], 100.0)
        self.assertEqual(res_won['predicted_winner'], 'Kolkata Knight Riders')

        # Edge Case 3: All out (10 wickets)
        res_out = predict_match_outcome('Punjab Kings', 'Rajasthan Royals', 'Jaipur', 200, 120, 15.0, 10)
        self.assertTrue(res_out['success'])
        self.assertEqual(res_out['batting_win_prob'], 0.0)
        self.assertEqual(res_out['bowling_win_prob'], 100.0)
        self.assertEqual(res_out['predicted_winner'], 'Rajasthan Royals')

        # Edge Case 4: Balls exhausted
        res_balls = predict_match_outcome('Sunrisers Hyderabad', 'Gujarat Titans', 'Ahmedabad', 170, 160, 20.0, 5)
        self.assertTrue(res_balls['success'])
        self.assertEqual(res_balls['bowling_win_prob'], 100.0)
        print("✓ Test 04 Passed: All boundary cricket scenarios validated.")

    def test_05_analytics_computations(self):
        """Verify analytics aggregations return non-empty expected structures."""
        matches = load_matches_data()
        kpis = get_overall_kpis(matches)
        self.assertEqual(kpis['total_matches'], 1243)
        self.assertEqual(kpis['total_seasons'], 18)

        stats = get_team_win_statistics(matches)
        self.assertGreater(len(stats), 10)
        self.assertIn('Win %', stats.columns)

        h2h = get_head_to_head_stats(matches, 'Chennai Super Kings', 'Mumbai Indians')
        self.assertGreater(h2h['total_matches'], 30)

        profile = get_team_detailed_profile(matches, 'Kolkata Knight Riders')
        self.assertGreater(profile['total_played'], 100)

        toss = get_toss_impact_analysis(matches)
        self.assertGreater(toss['toss_win_match_win_pct'], 40)

        venues = get_venue_analytics(matches)
        self.assertGreater(len(venues), 20)
        print("✓ Test 05 Passed: Analytics and metrics calculated successfully.")


if __name__ == '__main__':
    unittest.main()
