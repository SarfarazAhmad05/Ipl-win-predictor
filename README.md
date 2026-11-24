# 🏏 IPL Win Predictor & Sports Analytics Platform

A production-ready Machine Learning web application and sports analytics suite built with **Python**, **Streamlit**, and **Scikit-Learn**. The application integrates an existing pre-trained Machine Learning pipeline (`pipe.pkl`) to compute calibrated real-time winning probabilities during 2nd-innings IPL run chases.

---

## 🌟 Key Capabilities

1. **⚡ Dynamic In-Play Match Win Prediction (`pipe.pkl`):**
   - Live 2nd-innings chase forecasting based on real-time match state.
   - Evaluates required run rate (RRR), current run rate (CRR), wickets remaining, balls left, and host venue dynamics.
   - Outputs calibrated class probabilities via the trained Logistic Regression pipeline.
   - Interactive sensitivity analysis showing win probability variance if additional wickets fall.

2. **📊 Comprehensive IPL Historical Analytics:**
   - Visualizes records from **1,243 IPL matches** (2008 to Present) and **295,732 ball-by-ball deliveries**.
   - Team-wise winning percentages and all-time title/match leaders.
   - Venue profiles comparing batting first vs chasing success rates.
   - Toss impact analysis across all editions.

3. **⚔️ Franchise Profiles & Head-to-Head Comparison:**
   - In-depth profile for any IPL franchise.
   - Season-by-season win/loss trends and defend-versus-chase split.
   - Head-to-Head comparator between any two franchises with recent encounter match records.

4. **🔬 Complete Model Transparency & Evaluation:**
   - Full architecture diagram of `pipe.pkl` (ColumnTransformer with OneHotEncoder and Median Imputer).
   - Test evaluation results directly from `IPL.ipynb` (56.16% holdout accuracy).
   - Logistic regression coefficient analysis explaining feature importance in cricket terms.

5. **🎨 Premium Sports Analytics UI:**
   - Custom-crafted glassmorphism dark theme with IPL purple, electric blue, and orange accents.
   - Responsive layouts, custom KPI cards, and interactive Plotly visualizations.

---

## 📂 Project Structure

```text
ipl-win-predictor/
│
├── app.py                      # Application entry point & overview launchpad
├── IPL.ipynb                   # Original model training and preprocessing notebook
├── pipe.pkl                    # Trained Scikit-Learn pipeline (ColumnTransformer + LogisticRegression)
│
├── data/
│   ├── matches.csv             # Historical match summaries (1,243 rows)
│   └── deliveries.csv          # Ball-by-ball delivery telemetry (295,732 rows)
│
├── src/
│   ├── __init__.py
│   ├── prediction.py           # Pipeline loader, input validation, and probability inference
│   ├── data_loader.py          # Cached data loading and dataset cleaning utilities
│   ├── analytics.py            # Aggregations, KPIs, H2H metrics, and toss dynamics
│   └── ui_helpers.py           # Custom CSS injector, brand banners, and Plotly theme applicator
│
├── pages/
│   ├── 1_Home.py               # Page 1: Historical records, team wins, season trends
│   ├── 2_Match_Prediction.py   # Page 2: Core ML win predictor with sensitivity analysis
│   ├── 3_Team_Analytics.py     # Page 3: Franchise profiles and head-to-head records
│   ├── 4_Data_Analytics.py     # Page 4: Interactive Plotly macro analytics and venue insights
│   ├── 5_Model_Information.py  # Page 5: Model architecture, pipeline steps, and weights
│   └── 6_About.py              # Page 6: Problem statement, tech stack, and documentation
│
├── assets/
│   └── style.css               # Custom sports analytics dark glassmorphism stylesheet
│
├── requirements.txt            # Python dependencies
└── README.md                   # Complete documentation
```

---

## ⚙️ Model Architecture (`pipe.pkl`)

The pipeline expects 9 features in the following order:

| Feature | Type | Description |
|---|---|---|
| `batting_team` | Categorical | Chasing team (1 of 10 supported IPL franchises) |
| `bowling_team` | Categorical | Defending team (1 of 10 supported IPL franchises) |
| `city` | Categorical | Host city (34 supported cities + `Unknown` fallback) |
| `runs_left` | Numeric | Target runs minus current score |
| `balls_left` | Numeric | 120 minus balls completed |
| `wickets_left` | Numeric | 10 minus wickets lost |
| `runs_total_x` | Numeric | 1st innings total score set as target |
| `crr` | Numeric | Current Run Rate `(current_score * 6) / balls_bowled` |
| `rrr` | Numeric | Required Run Rate `(runs_left * 6) / balls_left` |

### Internal Pipeline Transformers
- **Categorical Sub-Pipeline:** `SimpleImputer(strategy='constant', fill_value='Unknown')` ➔ `OneHotEncoder(drop='first', handle_unknown='ignore')`
- **Numerical Sub-Pipeline:** `SimpleImputer(strategy='median')`
- **Classifier:** `LogisticRegression(solver='liblinear', C=1.0)`

---

## 🚀 Installation & Setup

### 1. Clone or Navigate to the Project Directory
```bash
cd C:\Users\sarfa\.gemini\antigravity\scratch\ipl-win-predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Or use the existing Python environment with `streamlit`, `scikit-learn`, `pandas`, `numpy`, `plotly`, and `joblib` installed).*

---

## 🎮 Running the Application

Launch the Streamlit web server:

```bash
streamlit run app.py
```

Once launched, open your web browser at:
```
http://localhost:8501
```

Use the left sidebar navigation to seamlessly transition between all 6 pages.
