import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle
import json
import os
from datetime import datetime
import subprocess
import sys

# --- SETUP & CONFIG ---
st.set_page_config(page_title="LaLiga Predictor Pro", page_icon="⚽", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1e2227; padding: 15px; border-radius: 10px; border: 1px solid #3e4451; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #e74c3c; color: white; border: none; font-weight: bold; }
    .stButton>button:hover { background-color: #c0392b; color: white; }
    .prediction-card { padding: 20px; border-radius: 15px; border-left: 5px solid #e74c3c; background: #1e2227; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

MODELS_DIR = "saved_models_pkl"
STATS_PATH = "current_team_stats.csv"
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")

# --- UTILS ---
def load_all_models():
    models = {}
    if not os.path.exists(MODELS_DIR):
        return models
    
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
    for f in files:
        # Initial guess at display name
        display_name = f.replace(".pkl", "").replace("_", " ").title().replace("Xgboost", "XGBoost").replace("Rf", "Random Forest").replace("Lgbm", "LightGBM").replace("Catboost", "CatBoost")
        
        # Cross-reference with model_metrics.json keys
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as mfile:
                metrics_data = json.load(mfile)
                for key in metrics_data.keys():
                    if key.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_') == f.replace(".pkl", ""):
                        display_name = key
                        break

        with open(os.path.join(MODELS_DIR, f), 'rb') as model_file:
            models[display_name] = {
                'model': pickle.load(model_file),
                'filename': f
            }
    return models

def sync_data():
    with st.status("📦 Syncing LaLiga Data & Retraining...", expanded=True) as status:
        st.write("Downloading Latest CSV from football-data.co.uk...")
        try:
            result = subprocess.run([sys.executable, "data2.py"], capture_output=True, text=True)
            if result.returncode == 0:
                st.write("✅ Data Integrated. Merged laliga_merged_clean.csv")
                st.write("✅ Elo Ratings Recalculated.")
                st.write("✅ Hybrid Models Retrained.")
                st.write("🏆 Top 3 models updated in saved_models_pkl/")
                status.update(label="Sync & Retrain Complete!", state="complete", expanded=False)
                st.rerun()
            else:
                st.error(f"Error during training: {result.stderr}")
        except Exception as e:
            st.error(f"Failed to trigger pipeline: {str(e)}")

def get_optimized_weights(display_name):
    clean_name = display_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
    weight_path = os.path.join(MODELS_DIR, f"{clean_name}_weights.json")
    if os.path.exists(weight_path):
        with open(weight_path, 'r') as f:
            return json.load(f)
    return {"model_w": 0.5, "elo_w": 0.5}

# --- DATA LOADING ---
if not os.path.exists(STATS_PATH):
    st.error("⚠️ Data stats missing! Please click 'Get Latest' in the sidebar to initialize the dataset.")
    if st.sidebar.button("📦 Initialize Dataset"):
        sync_data()
    st.stop()

team_stats = pd.read_csv(STATS_PATH)
teams = team_stats['Team'].unique()
all_models = load_all_models()

if not all_models:
    st.warning("⚠️ No trained models found in saved_models_pkl/. Running pipeline...")
    sync_data()

# --- LIVE ODDS LOADER ---
def load_live_odds():
    if os.path.exists("upcoming_odds.json"):
        with open("upcoming_odds.json", 'r') as f:
            return json.load(f)
    return []

def get_synced_odds(h_team, a_team, upcoming_data):
    # Mapping for common name variations between API and Football-Data.co.uk
    mapping = {
        'Real Sociedad': 'Sociedad',
        'Real Betis': 'Betis',
        'Atletico Madrid': 'Ath Madrid',
        'Athletic Bilbao': 'Ath Bilbao',
        'Celta Vigo': 'Celta',
        'Granada CF': 'Granada',
        'Deportivo Alaves': 'Alaves',
        'Alaves': 'Alaves',
        'Cadiz CF': 'Cadiz',
        'Rayo Vallecano': 'Vallecano',
        'UD Almeria': 'Almeria',
        'Espanyol': 'Espanol',
        'CA Osasuna': 'Osasuna',
        'Elche CF': 'Elche'
    }
    
    import unicodedata
    def normalize(name):
        if not name: return ""
        # Remove accents
        name = "".join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
        name = mapping.get(name, name)
        return name.lower().replace(' ', '').replace('-', '')

    for match in upcoming_data:
        api_h = normalize(match['Home'])
        api_a = normalize(match['Away'])
        sel_h = normalize(h_team)
        sel_a = normalize(a_team)
        
        # Check both home-away and away-home combinations
        if (api_h == sel_h and api_a == sel_a) or (api_h == sel_a and api_a == sel_h):
            return match
    return None

upcoming_odds_data = load_live_odds()

# --- SIDEBAR ---
with st.sidebar:
    st.image("LaLiga_EA_Sports_2023_Vertical_Logo.svg.png", width=100)
    st.title("Admin Controls")
    if st.button("📊 Get Latest Results & Retrain"):
        sync_data()
    
    st.divider()
    st.subheader("Model Selection")
    selected_model_name = st.selectbox("Predictive Engine", list(all_models.keys()))
    
    # Load specific metrics
    model_metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            model_metrics = json.load(f)
    
    metrics = model_metrics.get(selected_model_name, {"accuracy": 0.0, "log_loss": 0.0, "ml_weight": 0.5, "elo_weight": 0.5})
    
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    col2.metric("Log Loss", f"{metrics['log_loss']:.3f}")
    
    st.info(f"⚖️ Blended Analysis: {metrics.get('ml_weight', 0.5):.0%} ML / {metrics.get('elo_weight', 0.5):.0%} Elo")

# --- MAIN DASHBOARD ---
st.title("🇪🇸 LaLiga Match Predictor Pro")
st.markdown("### Next-Gen Hybrid Elo-ML Predictive Analytics")

tab1, tab2, tab3 = st.tabs(["🔮 Match Predictor", "📈 Deep Analytics", "🏆 Model Leaderboard"])

with tab1:
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("Home Team")
        home_team = st.selectbox("Select Home", teams, index=0)
        h_info = team_stats[team_stats['Team'] == home_team].iloc[0]
        st.write(f"**Elo Rating:** {h_info['Elo']:.0f}")
        st.write(f"**Recent Form:** {h_info['Form']} pts (last 5)")

    with col_r:
        st.subheader("Away Team")
        away_team = st.selectbox("Select Away", teams, index=1)
        a_info = team_stats[team_stats['Team'] == away_team].iloc[0]
        st.write(f"**Elo Rating:** {a_info['Elo']:.0f}")
        st.write(f"**Recent Form:** {a_info['Form']} pts (last 5)")

    # Check for Automated Odds
    matched_odds = get_synced_odds(home_team, away_team, upcoming_odds_data)
    
    st.divider()
    if matched_odds:
        st.success(f"✅ **Market Consensus Synced!** (Found in upcoming fixtures)")
        col1, col2, col3 = st.columns(3)
        m_h = col1.number_input("Market Home", value=matched_odds['Mean_H'])
        m_d = col2.number_input("Market Draw", value=matched_odds['Mean_D'])
        m_a = col3.number_input("Market Away", value=matched_odds['Mean_A'])
        st.caption(f"Sources aggregated: {matched_odds['Sources']} bookmakers (Mean Consensus)")
    else:
        st.warning("⚠️ No upcoming odds found for this pair. Using league averages.")
        col1, col2, col3 = st.columns(3)
        m_h = col1.number_input("Market Home", value=2.10)
        m_d = col2.number_input("Market Draw", value=3.30)
        m_a = col3.number_input("Market Away", value=3.50)

    if st.button("🔥 GENERATE PREDICTION"):
        # Calculate EloDiff (Home Adv Included: +75)
        elo_diff = (h_info['Elo'] + 75) - a_info['Elo']
        
        # ML Input Features
        input_data = pd.DataFrame([[
            elo_diff, h_info['GD'], a_info['GD'],
            m_h, m_d, m_a
        ]], columns=['EloDiff', 'H_GD', 'A_GD', 'Market_H', 'Market_D', 'Market_A'])
        
        # Get ML Probs
        model_obj = all_models[selected_model_name]['model']
        ml_probs = model_obj.predict_proba(input_data)[0] 
        
        # Get Elo Probs baseline
        def expected_result(elo_a, elo_b):
            return 1 / (10 ** ((elo_b - elo_a) / 400) + 1)
        
        elo_h_prob = expected_result(h_info['Elo'] + 75, a_info['Elo'])
        elo_a_prob = expected_result(a_info['Elo'], h_info['Elo'] + 75)
        elo_d_prob = 1 - (elo_h_prob + elo_a_prob)
        elo_probs = np.array([elo_h_prob, elo_d_prob, elo_a_prob])
        
        # Load optimized weights
        weights = get_optimized_weights(selected_model_name)
        w_ml = weights.get('model_w', 0.5)
        w_elo = weights.get('elo_w', 0.5)
        
        # Final Blend
        final_probs = (w_ml * ml_probs) + (w_elo * elo_probs)
        
        # Display Result
        results_labels = ["HOME WIN", "DRAW", "AWAY WIN"]
        winner_idx = np.argmax(final_probs)
        prob_val = final_probs[winner_idx]
        
        st.divider()
        st.balloons()
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='prediction-card'><h4>Verdict for {home_team} vs {away_team}:</h4><h1>{results_labels[winner_idx]}</h1><h2>{prob_val:.1%} Confidence</h2></div>", unsafe_allow_html=True)
        
        with c2:
            fig = go.Figure(go.Bar(
                x=final_probs,
                y=results_labels,
                orientation='h',
                marker_color=['#27ae60', '#f1c40f', '#e67e22']
            ))
            fig.update_layout(title="Probability Synthesis", template="plotly_dark", height=200, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
        with c3:
            st.subheader("Blended Intelligence")
            st.write(f"**ML Influence:** {w_ml:.0%}")
            st.write(f"**Elo Historical:** {w_elo:.0%}")
            st.progress(float(w_ml))

with tab2:
    st.subheader("Model Diagnostic Center")
    col_x, col_y = st.columns(2)
    
    with col_x:
        st.write("### 🎯 Confusion Matrix")
        if "confusion_matrix" in metrics:
            conf_m = np.array(metrics["confusion_matrix"])
            labels = ["Home", "Draw", "Away"]
            fig_cm = px.imshow(conf_m,
                               labels=dict(x="Predicted Outcome", y="Actual Outcome", color="Matches"),
                               x=labels, y=labels,
                               text_auto=True,
                               aspect="auto",
                               color_continuous_scale='Reds')
            fig_cm.update_layout(template="plotly_dark")
            st.plotly_chart(fig_cm, use_container_width=True)
            st.caption(f"Historical accuracy pattern for: {selected_model_name}")
        else:
            st.warning("Diagnostics missing. Please trigger a Re-Sync.")

    with col_y:
        st.write("### ⚖️ Feature/Baseline Split")
        split_data = pd.DataFrame({
            "Core": ["Machine Learning", "Elo Rating"],
            "Weight": [metrics.get('ml_weight', 0.5), metrics.get('elo_weight', 0.5)]
        })
        fig_pie = px.pie(split_data, values='Weight', names='Core', hole=0.5,
                         color_discrete_sequence=['#e74c3c', '#3498db'])
        fig_pie.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption(f"Optimized on test set for maximum generalization.")

    st.divider()
    st.write("### 📈 Team Performance Matrix")
    # Remove redundant market columns for a cleaner overview
    display_stats = team_stats.drop(columns=['Market_H', 'Market_D', 'Market_A'])
    st.dataframe(display_stats.sort_values("Elo", ascending=False), use_container_width=True)

with tab3:
    st.subheader("🏆 Global Model Leaderboard")
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            lb_metrics = json.load(f)
        
        rows = []
        for name, m in lb_metrics.items():
            rows.append({
                "Model Configuration": name,
                "Accuracy Score": f"{m['accuracy']:.2%}",
                "Log Loss Error": f"{m['log_loss']:.4f}",
                "Optimal ML Weight": f"{m.get('ml_weight', 0.5):.0%}",
                "Production Status": "🚀 Top Models (Active)" if name in all_models else "🔬 Retained Candidate"
            })
        
        leader_df = pd.DataFrame(rows).sort_values("Accuracy Score", ascending=False)
        st.table(leader_df)
        st.info("💡 Retraining occurs automatically whenever new match results are synced.")
    else:
        st.error("No metrics found. Please trigger a sync.")

st.divider()
st.caption(f"Interactive Analytics Hub | Verified Parity with data2.ipynb | Server Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
