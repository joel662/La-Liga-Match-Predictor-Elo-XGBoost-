import streamlit as st
import pandas as pd
import joblib
import json
import requests
import os
import plotly.graph_objects as go
from ensemble import EnsembleModel  # ensures module is loaded for unpickling

# ===============================
# CONFIG
# ===============================
DATA_URL = "https://www.football-data.co.uk/mmz4281/2526/SP1.csv"
DATA_DIR = "Match Data"
MERGED_FILE = "laliga_merged_clean.csv"
MODEL_DIR = "models_improved"
MODEL_TYPES = ["XGBoost", "Random Forest", "Ensemble"]

# ===============================
# MODERN MINIMAL CSS
# ===============================
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: linear-gradient(to bottom right, #0f172a, #1e293b);
    }
    
    /* Remove default padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Headers */
    h1 {
        font-weight: 700;
        font-size: 2.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    h2, h3 {
        font-weight: 600;
        color: #e2e8f0;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Subtitle */
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 3rem;
    }
    
    /* Glass card effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Stat cards */
    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.25rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        background: rgba(255, 255, 255, 0.08);
        transform: translateY(-2px);
    }
    
    .stat-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        filter: grayscale(30%);
    }
    
    .stat-label {
        color: #94a3b8;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .stat-value {
        color: #e2e8f0;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    /* Model metrics card */
    .metrics-container {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        padding: 1.5rem;
    }
    
    .metric-row {
        display: flex;
        gap: 2rem;
        justify-content: space-around;
    }
    
    .metric-item {
        text-align: center;
    }
    
    .metric-label-small {
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value-large {
        color: #e2e8f0;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    
    .metric-subtitle {
        color: #64748b;
        font-size: 0.875rem;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: #e2e8f0;
    }
    
    .stSelectbox label {
        color: #cbd5e1;
        font-weight: 500;
        font-size: 0.875rem;
    }
    
    /* Match display */
    .match-vs {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    
    .match-vs-text {
        color: #e2e8f0;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .vs-divider {
        color: #667eea;
        font-weight: 700;
        margin: 0 1rem;
    }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem;
    }
    
    div[data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
    }
    
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: #cbd5e1;
        font-weight: 500;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        margin: 3rem 0;
    }
    
    /* Alerts */
    .stAlert {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 3px solid #667eea;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    
    .info-box ul {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    
    .info-box li {
        color: #cbd5e1;
        margin: 0.5rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.875rem;
        margin-top: 4rem;
        padding-top: 2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ===============================
# HELPERS
# ===============================
def safe_last(series, default=0.0):
    series = series.dropna()
    if series.empty:
        return default
    return float(series.iloc[-1])


@st.cache_data
def load_base_artifacts():
    merged_df = pd.read_csv(MERGED_FILE)
    merged_df["Date"] = pd.to_datetime(merged_df["Date"], errors="coerce")
    last_update = merged_df["Date"].max()
    total_matches = len(merged_df)
    date_min = merged_df["Date"].min()
    date_max = merged_df["Date"].max()

    ratings_df = pd.read_csv(f"{MODEL_DIR}/current_elo_ratings.csv")
    feats_df = pd.read_csv(f"{MODEL_DIR}/training_features.csv")

    with open(f"{MODEL_DIR}/model_params.json") as f:
        params = json.load(f)

    feats_df["Date"] = pd.to_datetime(feats_df["Date"], errors="coerce")

    teams = ratings_df["Team"].unique().tolist()
    rows = []
    for team in teams:
        home_rows = feats_df[feats_df["HomeTeam"] == team].sort_values("Date")
        away_rows = feats_df[feats_df["AwayTeam"] == team].sort_values("Date")

        row = {
            "Team": team,
            "HomeFormPts5": safe_last(home_rows["HomeFormPts5"]) if "HomeFormPts5" in feats_df.columns else 0.0,
            "AwayFormPts5": safe_last(away_rows["AwayFormPts5"]) if "AwayFormPts5" in feats_df.columns else 0.0,
            "HomeGD5": safe_last(home_rows["HomeGD5"]) if "HomeGD5" in feats_df.columns else 0.0,
            "AwayGD5": safe_last(away_rows["AwayGD5"]) if "AwayGD5" in feats_df.columns else 0.0,
            "HomeRestDays": safe_last(home_rows["HomeRestDays"]) if "HomeRestDays" in feats_df.columns else 0.0,
            "AwayRestDays": safe_last(away_rows["AwayRestDays"]) if "AwayRestDays" in feats_df.columns else 0.0,
        }
        rows.append(row)

    team_stats = pd.DataFrame(rows)
    return ratings_df, feats_df, params, last_update, team_stats, total_matches, date_min, date_max


@st.cache_data
def load_model(model_choice: str):
    if model_choice == "XGBoost":
        model = joblib.load(f"{MODEL_DIR}/xgb_model.pkl")
    elif model_choice == "Random Forest":
        model = joblib.load(f"{MODEL_DIR}/rf_model.pkl")
    else:
        model = joblib.load(f"{MODEL_DIR}/ensemble_model.pkl")
    return model

# ===============================
# MAIN APP
# ===============================
st.set_page_config(
    page_title="La Liga Predictor", 
    page_icon="⚽", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

load_custom_css()

# Header
st.markdown("# La Liga Match Predictor")
st.markdown('<p class="subtitle">Advanced Elo rating system combined with machine learning for accurate match predictions</p>', unsafe_allow_html=True)

# Load data
ratings_df, feats_df, params, last_update, team_stats, total_matches, date_min, date_max = load_base_artifacts()

# Stats bar
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">📊</div>
        <div class="stat-label">Total Matches</div>
        <div class="stat-value">{total_matches:,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">📅</div>
        <div class="stat-label">Last Updated</div>
        <div class="stat-value">{last_update.strftime('%Y-%m-%d') if pd.notna(last_update) else 'Unknown'}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">📈</div>
        <div class="stat-label">Date Range</div>
        <div class="stat-value" style="font-size: 1rem;">{date_min.strftime('%Y')} - {date_max.strftime('%Y')}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Model selection
st.markdown("## Model Selection")

col_select, col_metrics = st.columns([1, 2])

with col_select:
    model_choice = st.selectbox("Choose model", MODEL_TYPES, label_visibility="visible")

with col_metrics:
    metrics = params.get("metrics", {})
    
    if model_choice in metrics:
        model_metrics = metrics[model_choice]
        acc = model_metrics.get("accuracy", 0)
        ll = model_metrics.get("logloss", 0)
        
        st.markdown(f"""
        <div class="metrics-container">
            <div class="metric-row">
                <div class="metric-item">
                    <div class="metric-label-small">Accuracy</div>
                    <div class="metric-value-large">{acc:.4f}</div>
                    <div class="metric-subtitle">{acc*100:.2f}%</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label-small">Log Loss</div>
                    <div class="metric-value-large">{ll:.4f}</div>
                    <div class="metric-subtitle">lower is better</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info(f"No metrics available for {model_choice}")

model = load_model(model_choice)

st.markdown("<hr>", unsafe_allow_html=True)

# Team selection
st.markdown("## Match Setup")

col1, col2 = st.columns(2)
home_team = col1.selectbox("Home Team", ratings_df["Team"].tolist(), label_visibility="visible")
away_team = col2.selectbox("Away Team", ratings_df["Team"].tolist(), label_visibility="visible")

if home_team == away_team:
    st.warning("Please select different teams")
    st.stop()

st.markdown(f'<div class="match-vs"><span class="match-vs-text">{home_team}</span><span class="vs-divider">vs</span><span class="match-vs-text">{away_team}</span></div>', unsafe_allow_html=True)

# Feature construction
home_elo = float(ratings_df.loc[ratings_df["Team"] == home_team, "Elo"].iloc[0])
away_elo = float(ratings_df.loc[ratings_df["Team"] == away_team, "Elo"].iloc[0])
elo_diff = home_elo - away_elo + params["HOME_ADV"]

home_stats = team_stats.loc[team_stats["Team"] == home_team].iloc[0]
away_stats = team_stats.loc[team_stats["Team"] == away_team].iloc[0]

feature_cols = params.get(
    "feature_cols",
    ["EloHomeBefore", "EloAwayBefore", "EloDiff", "HomeFormPts5", "AwayFormPts5", "HomeGD5", "AwayGD5", "HomeRestDays", "AwayRestDays"]
)

features = {}
for col in feature_cols:
    if col == "EloHomeBefore":
        features[col] = home_elo
    elif col == "EloAwayBefore":
        features[col] = away_elo
    elif col == "EloDiff":
        features[col] = elo_diff
    elif col == "HomeFormPts5":
        features[col] = float(home_stats["HomeFormPts5"])
    elif col == "AwayFormPts5":
        features[col] = float(away_stats["AwayFormPts5"])
    elif col == "HomeGD5":
        features[col] = float(home_stats["HomeGD5"])
    elif col == "AwayGD5":
        features[col] = float(away_stats["AwayGD5"])
    elif col == "HomeRestDays":
        features[col] = float(home_stats["HomeRestDays"])
    elif col == "AwayRestDays":
        features[col] = float(home_stats["AwayRestDays"])
    else:
        features[col] = 0.0

X_new = pd.DataFrame([[features[c] for c in feature_cols]], columns=feature_cols)

# Predictions
st.markdown("## Prediction Results")

probs = model.predict_proba(X_new)[0]

col1, col2, col3 = st.columns(3)
col1.metric("Home Win", f"{probs[0]*100:.1f}%")
col2.metric("Draw", f"{probs[1]*100:.1f}%")
col3.metric("Away Win", f"{probs[2]*100:.1f}%")

# Chart
labels = ["Home Win", "Draw", "Away Win"]
colors = ["#667eea", "#764ba2", "#f093fb"]

fig = go.Figure(
    go.Bar(
        x=probs * 100,
        y=labels,
        orientation="h",
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="inside",
        marker_color=colors,
        textfont=dict(size=14, color="white", family="Inter"),
    )
)
fig.update_layout(
    xaxis_title="Probability (%)",
    height=300,
    plot_bgcolor="rgba(15, 23, 42, 0.5)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cbd5e1", size=12, family="Inter"),
    xaxis=dict(gridcolor="rgba(255, 255, 255, 0.1)"),
    yaxis=dict(gridcolor="rgba(255, 255, 255, 0.1)"),
    margin=dict(l=20, r=20, t=20, b=40),
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("View feature details"):
    st.dataframe(X_new.T.rename(columns={0: "value"}), use_container_width=True)

# Elo history
st.markdown("## Elo Rating History")

home_hist = feats_df[feats_df["HomeTeam"] == home_team][["Date", "EloHomeBefore"]]
away_hist = feats_df[feats_df["AwayTeam"] == away_team][["Date", "EloAwayBefore"]]

trend = pd.concat([
    home_hist.rename(columns={"EloHomeBefore": "Elo"}).assign(Team=home_team),
    away_hist.rename(columns={"EloAwayBefore": "Elo"}).assign(Team=away_team),
])

fig2 = go.Figure()
for i, (team, sub) in enumerate(trend.groupby("Team")):
    color = ["#667eea", "#764ba2"][i]
    fig2.add_trace(
        go.Scatter(
            x=sub["Date"],
            y=sub["Elo"],
            mode="lines",
            name=team,
            line=dict(width=3, color=color),
            fill='tonexty' if i > 0 else None,
        )
    )

fig2.update_layout(
    yaxis_title="Elo Rating",
    height=400,
    plot_bgcolor="rgba(15, 23, 42, 0.5)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cbd5e1", size=12, family="Inter"),
    xaxis=dict(gridcolor="rgba(255, 255, 255, 0.1)"),
    yaxis=dict(gridcolor="rgba(255, 255, 255, 0.1)"),
    legend=dict(
        bgcolor="rgba(255, 255, 255, 0.05)",
        bordercolor="rgba(255, 255, 255, 0.1)",
        borderwidth=1
    ),
    margin=dict(l=20, r=20, t=20, b=40),
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Data refresh
st.markdown("## Update Models")

st.markdown("""
<div class="info-box">
<strong>This will:</strong>
<ul>
<li>Download the latest La Liga data</li>
<li>Merge and clean all season files</li>
<li>Retrain all machine learning models</li>
<li>Update the ensemble predictor</li>
</ul>
</div>
""", unsafe_allow_html=True)

if st.button("Fetch Latest Data & Retrain"):
    st.info("Downloading latest data...")
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, "SP1_latest.csv")

    try:
        resp = requests.get(DATA_URL, timeout=30)
        resp.raise_for_status()
        with open(csv_path, "wb") as f:
            f.write(resp.content)
        st.success("Data downloaded successfully")
    except Exception as e:
        st.error(f"Download failed: {e}")
        st.stop()

    st.info("Merging datasets...")
    if os.system("python merge.py") != 0:
        st.error("Merge failed")
        st.stop()
    st.success("Datasets merged")

    st.info("Training models...")
    if os.system(f"python train_improved.py {MERGED_FILE}") != 0:
        st.error("Training failed")
        st.stop()
    st.success("Models trained")

    st.info("Building ensemble...")
    if os.system("python ensemble.py") != 0:
        st.error("Ensemble build failed")
        st.stop()
    st.success("All models updated successfully!")
    st.warning("Please refresh the page to load new models")

st.markdown('<div class="footer">Developed by Joel • La Liga Match Prediction System</div>', unsafe_allow_html=True)