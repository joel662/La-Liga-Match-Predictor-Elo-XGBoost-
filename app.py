import streamlit as st
import pandas as pd
import joblib
import json
import plotly.graph_objects as go
from datetime import datetime

# ---------- Load ----------
@st.cache_data
def load_artifacts():
    ratings = pd.read_csv("models/current_elo_ratings.csv")
    feats = pd.read_csv("models/training_features.csv")
    with open("models/model_params.json") as f:
        params = json.load(f)
    model = joblib.load("models/xgb_model.pkl")
    return ratings, feats, params, model

ratings_df, feats_df, params, model = load_artifacts()

# ---------- UI ----------
st.title("⚽ La Liga Match Predictor (Elo + XGBoost)")
st.markdown("### Hybrid probabilistic model with Elo features and XGBoost learning")

col1, col2 = st.columns(2)
home_team = col1.selectbox("🏠 Home Team", ratings_df["Team"].tolist())
away_team = col2.selectbox("🛫 Away Team", ratings_df["Team"].tolist())
if home_team == away_team:
    st.warning("Please select two different teams.")
    st.stop()

home_elo = float(ratings_df.loc[ratings_df["Team"]==home_team,"Elo"].iloc[0])
away_elo = float(ratings_df.loc[ratings_df["Team"]==away_team,"Elo"].iloc[0])
elo_diff = home_elo - away_elo + params["HOME_ADV"]

X_new = pd.DataFrame([[home_elo, away_elo, elo_diff]], columns=["EloHomeBefore","EloAwayBefore","EloDiff"])
probs = model.predict_proba(X_new)[0]
labels = ["Home Win","Draw","Away Win"]

st.subheader("📊 Win Probabilities")
fig = go.Figure(go.Bar(
    x=probs*100, y=labels, orientation="h",
    text=[f"{p*100:.1f}%" for p in probs],
    textposition="auto",
    marker_color=["#4CAF50","#FFC107","#F44336"]
))
fig.update_layout(xaxis_title="Probability (%)", height=300)
st.plotly_chart(fig, use_container_width=True)

# ---------- Elo Trend ----------
st.subheader("📈 Elo History")
home_trend = feats_df[feats_df["HomeTeam"]==home_team][["Date","EloHomeBefore"]]
away_trend = feats_df[feats_df["AwayTeam"]==away_team][["Date","EloAwayBefore"]]
trend = pd.concat([
    home_trend.rename(columns={"EloHomeBefore":"Elo"}).assign(Team=home_team),
    away_trend.rename(columns={"EloAwayBefore":"Elo"}).assign(Team=away_team)
])
trend["Date"] = pd.to_datetime(trend["Date"])
fig2 = go.Figure()
for team, sub in trend.groupby("Team"):
    fig2.add_trace(go.Scatter(x=sub["Date"], y=sub["Elo"], mode="lines+markers", name=team))
fig2.update_layout(yaxis_title="Elo Rating", height=400)
st.plotly_chart(fig2, use_container_width=True)

# ---------- Retrain ----------
if st.button("🔁 Retrain Model"):
    import os, subprocess
    st.info("Retraining model... this may take a minute.")
    os.system("python train.py laliga_merged_clean.csv")
    st.success("Model retrained. Reload the page to see updates.")

st.caption("Developed by Joel — Hybrid Elo + XGBoost Match Prediction System")
