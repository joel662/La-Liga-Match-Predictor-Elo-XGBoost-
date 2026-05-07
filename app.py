"""
⚽ Multi-League Match Predictor — Streamlit App
================================================
Upload your all_leagues.csv, select a league scenario,
pick a model, and predict any match.
"""

import os
import sys
import json
import pickle
import warnings
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚽ Match Predictor",
    page_icon="⚽",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }
    .stButton > button {
        background: linear-gradient(135deg, #1db954, #128040);
        color: white; font-weight: 700; border: none;
        border-radius: 8px; padding: 0.6rem 2rem;
        font-size: 1rem; transition: 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; transform: scale(1.02); }
    .pred-card {
        background: #1e2130; border-radius: 12px;
        padding: 1.5rem; text-align: center;
        border: 1px solid #2e3250;
    }
    .pred-title { font-size: 1rem; color: #888; margin-bottom: 0.4rem; }
    .pred-value { font-size: 2.2rem; font-weight: 800; }
    .pred-pct   { font-size: 1rem; color: #aaa; margin-top: 0.2rem; }
    .winner-card {
        background: linear-gradient(135deg, #1a3a2a, #162e1e);
        border: 2px solid #1db954; border-radius: 14px;
        padding: 1.5rem; text-align: center;
    }
    h1 { color: #1db954 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HAS_GPU  = DEVICE.type == "cuda"
CUTOFF   = pd.Timestamp("2025-08-01")
TSCV     = TimeSeriesSplit(n_splits=3)
TOP5_DIV = ["E0", "SP1", "I1", "D1", "F1"]
HIST_LEN = 10
PTS_MAP  = {"H": 3, "D": 1, "A": 0}
FEATURE_COLS = [
    "EloDiff", "EloCloseness", "H_Form", "A_Form", "H_Form10", "A_Form10",
    "H_Momentum", "A_Momentum", "H_GD", "A_GD", "H_Attack", "A_Attack",
    "H_Defence", "A_Defence", "H_Score_Std", "A_Score_Std", "AvgGoalsExpected",
    "H_CS_Rate", "A_CS_Rate", "DrawTendency", "H_Home_Form", "A_Away_Form",
    "H_Rest", "A_Rest", "H_Recent_Games", "A_Recent_Games",
    "H2H_Win_Rate", "H2H_Draw_Rate", "Market_H", "Market_D", "Market_A",
    "Imp_H", "Imp_D", "Imp_A", "Imp_H_minus_A", "Imp_D_sq", "Elo_x_ImpH",
    "Div_Code", "League_Draw_Rate",
]

SCENARIOS = {
    "Premier League":        ["E0"],
    "La Liga":               ["SP1"],
    "Serie A":               ["I1"],
    "Bundesliga":            ["D1"],
    "Ligue 1":               ["F1"],
    "Top 5 Leagues Combined": TOP5_DIV,
    "All Leagues Combined":  None,
}

MODEL_OPTIONS = [
    "Logistic Regression", "Random Forest", "Linear SVM",
    "XGBoost", "LightGBM", "CatBoost", "MLP (PyTorch)", "HistGBM",
    "Stacked (CatMeta)", "Stacked (MLPMeta)", "Stacked (RFMeta)", "Stacked (XGBMeta)",
    "Hybrid (XGBoost-Elo)", "Hybrid (LightGBM-Elo)", "Hybrid (CatBoost-Elo)",
    "Hybrid (Random Forest-Elo)", "Best Auto-Selected",
]

# ─────────────────────────────────────────────────────────────────────────────
# PyTorch MLP
# ─────────────────────────────────────────────────────────────────────────────
class TorchMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(64, 3),
        )
    def forward(self, x):
        return self.net(x)


def train_torch_mlp(X_train, y_train, epochs=30, batch_size=1024):
    from torch.utils.data import DataLoader, TensorDataset
    import torch.optim as optim

    X_np = X_train.values.astype(np.float32)
    y_np = y_train.values.astype(np.int64)
    counts = np.bincount(y_np, minlength=3)
    cw = torch.tensor(len(y_np) / (3 * counts + 1e-8), dtype=torch.float32).to(DEVICE)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_np), torch.from_numpy(y_np)),
        batch_size=batch_size, shuffle=True, pin_memory=HAS_GPU, num_workers=0,
    )
    model     = TorchMLP(X_train.shape[1]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-3, steps_per_epoch=len(loader), epochs=epochs)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=HAS_GPU)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx = bx.to(DEVICE, non_blocking=True)
            by = by.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=HAS_GPU):
                loss = criterion(model(bx), by)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            scheduler.step()
    return model


def predict_torch_mlp(model, X):
    model.eval()
    X_np = X.values.astype(np.float32) if hasattr(X, "values") else np.asarray(X, dtype=np.float32)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=HAS_GPU):
        t = torch.from_numpy(X_np).to(DEVICE, non_blocking=True)
        return torch.softmax(model(t), dim=1).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Elo helpers
# ─────────────────────────────────────────────────────────────────────────────
def expected_result(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def update_elo_mov(elo_home, elo_away, result, goal_diff, k=20, home_adv=75):
    exp_home   = expected_result(elo_home + home_adv, elo_away)
    score_home = 1 if result == 1 else (0.5 if result == 0 else 0)
    mov        = min(np.log(abs(goal_diff) + 1) + 1.0, 2.5)
    k_eff      = k * mov
    return (
        elo_home + k_eff * (score_home - exp_home),
        elo_away + k_eff * ((1 - score_home) - (1 - exp_home)),
    )

def get_elo_win_probs_vec(elo_h, elo_a, home_adv=75, draw_prob=0.25):
    win_h = 1 / (1 + 10 ** ((elo_a - (elo_h + home_adv)) / 400))
    win_a = 1 / (1 + 10 ** (((elo_h + home_adv) - elo_a) / 400))
    rem   = 1.0 - draw_prob
    p_d   = np.full_like(win_h, draw_prob, dtype=float)
    return np.column_stack([win_h * rem, p_d, win_a * rem])

def get_pts(dq, n=5):
    return sum(PTS_MAP.get(r, 0) for r in list(dq)[-n:])

def _make_hist():
    return {k: deque(maxlen=HIST_LEN) for k in
            ("dates", "results", "goals_for", "goals_against",
             "home_results", "away_results", "clean_sheets")}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & feature engineering (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="🔧 Building features (runs once)…")
def load_and_engineer(file_bytes: bytes):
    import io
    df = pd.read_csv(io.BytesIO(file_bytes))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Odds imputation
    for suffix in ["H", "D", "A"]:
        for prefix in ["PS", "BbAv"]:
            col, ref = prefix + suffix, "B365" + suffix
            if col in df.columns and ref in df.columns:
                df[col] = df[col].fillna(df[ref])

    for suffix in ["H", "D", "A"]:
        cols = [c + suffix for c in ["B365", "BbAv", "PS"] if c + suffix in df.columns]
        df[f"Market_{suffix}"] = df[cols].mean(axis=1)

    # Feature engineering loop
    teams    = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    elo      = {t: 1500.0 for t in teams}
    team_hist = {t: _make_hist() for t in teams}
    h2h: dict = {}
    feature_rows = []

    for _, row in df.iterrows():
        home, away, date = row["HomeTeam"], row["AwayTeam"], row["Date"]
        try:
            hg, ag = int(row["FTHG"]), int(row["FTAG"])
        except (ValueError, TypeError):
            continue
        res_label = row["FTR"]
        res       = 1 if res_label == "H" else (0 if res_label == "D" else -1)
        goal_diff = hg - ag

        elo_h, elo_a = elo[home], elo[away]
        elo_diff     = (elo_h + 75) - elo_a
        elo_close    = 1 / (1 + abs(elo_diff) / 100)

        hh, ah = team_hist[home], team_hist[away]
        h_res5, a_res5 = list(hh["results"])[-5:], list(ah["results"])[-5:]
        h_form         = get_pts(hh["results"], 5)
        a_form         = get_pts(ah["results"], 5)
        h_form10       = get_pts(hh["results"], 10)
        a_form10       = get_pts(ah["results"], 10)
        draw_tendency  = (h_res5.count("D") / max(len(h_res5), 1)
                          + a_res5.count("D") / max(len(a_res5), 1)) / 2

        h_gf5 = list(hh["goals_for"])[-5:]
        a_gf5 = list(ah["goals_for"])[-5:]
        h_ga5 = list(hh["goals_against"])[-5:]
        a_ga5 = list(ah["goals_against"])[-5:]
        h_gd, a_gd = sum(h_gf5) - sum(h_ga5), sum(a_gf5) - sum(a_ga5)

        hgf = list(hh["goals_for"]);  hga = list(hh["goals_against"])
        agf = list(ah["goals_for"]);  aga = list(ah["goals_against"])

        h_attack  = np.mean(hgf[-10:]) if hgf else 1.2
        h_defence = np.mean(hga[-10:]) if hga else 1.1
        a_attack  = np.mean(agf[-10:]) if agf else 1.0
        a_defence = np.mean(aga[-10:]) if aga else 1.2

        h_score_std = np.std(hgf[-10:]) if len(hgf) >= 3 else 1.0
        a_score_std = np.std(agf[-10:]) if len(agf) >= 3 else 1.0
        h_cs_rate   = np.mean(list(hh["clean_sheets"])[-10:]) if hh["clean_sheets"] else 0.25
        a_cs_rate   = np.mean(list(ah["clean_sheets"])[-10:]) if ah["clean_sheets"] else 0.20

        h_dates, a_dates = list(hh["dates"]), list(ah["dates"])
        h_rest   = int((date - h_dates[-1]) / pd.Timedelta(days=1)) if h_dates else 14
        a_rest   = int((date - a_dates[-1]) / pd.Timedelta(days=1)) if a_dates else 14
        h_recent = sum(1 for d in h_dates[-8:] if int((date - d) / pd.Timedelta(days=1)) <= 14)
        a_recent = sum(1 for d in a_dates[-8:] if int((date - d) / pd.Timedelta(days=1)) <= 14)

        h2h_hist      = h2h.get((home, away), [])[-5:]
        h2h_win_rate  = h2h_hist.count("H") / max(len(h2h_hist), 1)
        h2h_draw_rate = h2h_hist.count("D") / max(len(h2h_hist), 1)

        h_home_form = get_pts(hh["home_results"], 5)
        a_away_form = get_pts(ah["away_results"], 5)
        h_form3     = get_pts(hh["results"], 3)
        a_form3     = get_pts(ah["results"], 3)
        h_momentum  = h_form3 / 9 - (h_form10 / 30 if h_form10 else 0)
        a_momentum  = a_form3 / 9 - (a_form10 / 30 if a_form10 else 0)

        feature_rows.append({
            "Date": date, "HomeTeam": home, "AwayTeam": away,
            "EloH": elo_h, "EloA": elo_a,
            "EloDiff": elo_diff, "EloCloseness": elo_close,
            "H_Form": h_form,    "A_Form": a_form,
            "H_Form10": h_form10, "A_Form10": a_form10,
            "H_Momentum": h_momentum, "A_Momentum": a_momentum,
            "H_GD": h_gd, "A_GD": a_gd,
            "H_Attack": h_attack,  "A_Attack": a_attack,
            "H_Defence": h_defence, "A_Defence": a_defence,
            "H_Score_Std": h_score_std, "A_Score_Std": a_score_std,
            "AvgGoalsExpected": (np.mean(h_gf5) if h_gf5 else 1.2) + (np.mean(a_gf5) if a_gf5 else 1.0),
            "H_CS_Rate": h_cs_rate, "A_CS_Rate": a_cs_rate,
            "DrawTendency": draw_tendency,
            "H_Home_Form": h_home_form, "A_Away_Form": a_away_form,
            "H_Rest": min(h_rest, 20), "A_Rest": min(a_rest, 20),
            "H_Recent_Games": h_recent, "A_Recent_Games": a_recent,
            "H2H_Win_Rate": h2h_win_rate, "H2H_Draw_Rate": h2h_draw_rate,
            "Market_H": row.get("Market_H", np.nan),
            "Market_D": row.get("Market_D", np.nan),
            "Market_A": row.get("Market_A", np.nan),
            "Target": 0 if res == 1 else (1 if res == 0 else 2),
        })

        res_str = "H" if res == 1 else ("D" if res == 0 else "A")
        for t, gf, ga, is_home in [(home, hg, ag, True), (away, ag, hg, False)]:
            pov = res_str if is_home else ("A" if res_str == "H" else ("H" if res_str == "A" else "D"))
            hst = team_hist[t]
            hst["dates"].append(date)
            hst["results"].append(pov)
            hst["goals_for"].append(gf)
            hst["goals_against"].append(ga)
            hst["clean_sheets"].append(1 if ga == 0 else 0)
            (hst["home_results"] if is_home else hst["away_results"]).append(pov)
        h2h.setdefault((home, away), []).append(res_str)
        elo[home], elo[away] = update_elo_mov(elo_h, elo_a, res, goal_diff)

    feats_df = pd.DataFrame(feature_rows)

    df["Div_Code"]         = df["Div"].astype("category").cat.codes
    league_draw_rates      = df.groupby("Div")["FTR"].apply(lambda x: (x == "D").mean())
    df["League_Draw_Rate"] = df["Div"].map(league_draw_rates)
    feats_df["Div_Code"]         = df["Div_Code"].values
    feats_df["League_Draw_Rate"] = df["League_Draw_Rate"].values

    total_inv = (1 / feats_df["Market_H"]) + (1 / feats_df["Market_D"]) + (1 / feats_df["Market_A"])
    feats_df["Imp_H"]         = (1 / feats_df["Market_H"]) / total_inv
    feats_df["Imp_D"]         = (1 / feats_df["Market_D"]) / total_inv
    feats_df["Imp_A"]         = (1 / feats_df["Market_A"]) / total_inv
    feats_df["Imp_H_minus_A"] = feats_df["Imp_H"] - feats_df["Imp_A"]
    feats_df["Imp_D_sq"]      = feats_df["Imp_D"] ** 2
    feats_df["Elo_x_ImpH"]    = feats_df["EloDiff"] * feats_df["Imp_H"]

    feats_df[FEATURE_COLS] = (feats_df[FEATURE_COLS]
                               .fillna(feats_df[FEATURE_COLS].mean())
                               .fillna(0))

    return df, feats_df, elo, team_hist


# ─────────────────────────────────────────────────────────────────────────────
# Model training (cached per scenario)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Training models (first run only, ~2–5 min)…")
def train_models(file_hash: str, scenario_name: str, div_filter_key: str,
                 feats_df_json: str, df_json: str):
    feats_df = pd.read_json(feats_df_json)
    feats_df["Date"] = pd.to_datetime(feats_df["Date"])
    df_raw = pd.read_json(df_json)
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])

    div_filter = SCENARIOS[scenario_name]

    if div_filter:
        mask      = df_raw["Div"].isin(div_filter)
        cur_feats = feats_df[mask.values[:len(feats_df)]].copy()
    else:
        cur_feats = feats_df.copy()

    feats_train = cur_feats[cur_feats["Date"] < CUTOFF].copy()
    if len(feats_train) < 300:
        return None, "Too few training rows for this scenario."

    X_all = feats_train[FEATURE_COLS]
    y_all = feats_train["Target"]

    XGB_GPU = HAS_GPU
    LGB_GPU = HAS_GPU
    CAT_GPU = HAS_GPU

    base_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, n_jobs=-1),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced",
                                                       min_samples_leaf=3, random_state=42, n_jobs=-1),
        "Linear SVM":          CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced", max_iter=2000)),
        "XGBoost":             XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                              objective="multi:softprob", num_class=3,
                                              eval_metric="mlogloss", verbosity=0,
                                              tree_method="hist",
                                              device="cuda" if XGB_GPU else "cpu"),
        "LightGBM":            LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=40,
                                               class_weight="balanced", random_state=42, verbose=-1,
                                               device="gpu" if LGB_GPU else "cpu"),
        "CatBoost":            CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6,
                                                   verbose=False, random_state=42,
                                                   task_type="GPU" if CAT_GPU else "CPU"),
        "MLP":                 None,
        "HistGBM":             HistGradientBoostingClassifier(max_iter=150, max_depth=6,
                                                               learning_rate=0.05,
                                                               class_weight="balanced", random_state=42),
    }
    model_names = list(base_models.keys())
    n_models    = len(model_names)

    # OOF generation
    oof_meta = np.zeros((len(X_all), n_models * 3))
    tscv_inner = TimeSeriesSplit(n_splits=3)
    for fold_i, (tr_idx, val_idx) in enumerate(tscv_inner.split(X_all)):
        Xtr, Xval = X_all.iloc[tr_idx], X_all.iloc[val_idx]
        ytr        = y_all.iloc[tr_idx]
        sc     = StandardScaler()
        Xtr_sc = pd.DataFrame(sc.fit_transform(Xtr), columns=Xtr.columns)
        Xval_sc = pd.DataFrame(sc.transform(Xval), columns=Xval.columns)
        for m_idx, (name, model) in enumerate(base_models.items()):
            cs, ce = m_idx * 3, m_idx * 3 + 3
            try:
                if name == "MLP":
                    m = train_torch_mlp(Xtr_sc, ytr)
                    p = predict_torch_mlp(m, Xval_sc)
                elif name == "XGBoost":
                    sw = compute_sample_weight("balanced", ytr)
                    model.fit(Xtr, ytr, sample_weight=sw)
                    p = model.predict_proba(Xval)
                else:
                    model.fit(Xtr, ytr)
                    p = model.predict_proba(Xval)
                oof_meta[val_idx, cs:ce] = p
            except Exception:
                oof_meta[val_idx, cs:ce] = 1/3

    # Full retrain
    sc_full  = StandardScaler()
    X_all_sc = pd.DataFrame(sc_full.fit_transform(X_all), columns=X_all.columns)
    torch_full_model = None

    for name, model in base_models.items():
        try:
            if name == "MLP":
                torch_full_model = train_torch_mlp(X_all_sc, y_all)
            elif name == "XGBoost":
                sw = compute_sample_weight("balanced", y_all)
                model.fit(X_all, y_all, sample_weight=sw)
            else:
                model.fit(X_all, y_all)
        except Exception:
            pass

    # Meta-learners
    oof_aug = np.hstack([oof_meta, X_all.values])
    cat_meta = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6,
                                   verbose=False, random_state=42,
                                   task_type="GPU" if CAT_GPU else "CPU")
    cat_meta.fit(oof_aug, y_all)

    mlp_meta_sc = StandardScaler()
    mlp_meta    = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                                 max_iter=300, early_stopping=True, random_state=42)
    mlp_meta.fit(mlp_meta_sc.fit_transform(oof_aug), y_all)

    rf_meta  = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf_meta.fit(oof_aug, y_all)

    xgb_meta = XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=4,
                               eval_metric="mlogloss", verbosity=0,
                               tree_method="hist", device="cuda" if XGB_GPU else "cpu",
                               random_state=42)
    xgb_meta.fit(oof_aug, y_all)

    # Hybrid weight optimisation (quick version)
    tscv_hw = TimeSeriesSplit(n_splits=3)
    splits  = list(tscv_hw.split(X_all))
    tr_last, val_last = splits[-1]
    X_val_hw = X_all.iloc[val_last]
    y_val_hw = y_all.iloc[val_last].values
    sc_hw    = StandardScaler()
    X_tr_sc_hw = pd.DataFrame(sc_hw.fit_transform(X_all.iloc[tr_last]), columns=X_all.columns)
    X_val_sc_hw = pd.DataFrame(sc_hw.transform(X_val_hw), columns=X_all.columns)

    elo_probs_val = get_elo_win_probs_vec(
        feats_train.iloc[val_last]["EloH"].values,
        feats_train.iloc[val_last]["EloA"].values,
    )
    weights_grid  = np.round(np.arange(0.0, 1.01, 0.05), 2)
    hybrid_weights: dict = {}
    for name, model in base_models.items():
        try:
            if name == "MLP":
                m = train_torch_mlp(X_tr_sc_hw, y_all.iloc[tr_last])
                base_p = predict_torch_mlp(m, X_val_sc_hw)
            elif name == "XGBoost":
                sw = compute_sample_weight("balanced", y_all.iloc[tr_last])
                model.fit(X_all.iloc[tr_last], y_all.iloc[tr_last], sample_weight=sw)
                base_p = model.predict_proba(X_val_hw)
            else:
                model.fit(X_all.iloc[tr_last], y_all.iloc[tr_last])
                base_p = model.predict_proba(X_val_hw)
            blended = (weights_grid[:, None, None] * base_p[None]
                       + (1 - weights_grid)[:, None, None] * elo_probs_val[None])
            f1s    = np.array([f1_score(y_val_hw, np.argmax(b, axis=1), average="macro")
                                for b in blended])
            best_w = float(weights_grid[np.argmax(f1s)])
            hybrid_weights[name] = {"model_w": best_w, "elo_w": round(1 - best_w, 2)}
        except Exception:
            hybrid_weights[name] = {"model_w": 0.7, "elo_w": 0.3}

    # Retrain on full data with fresh OOF for meta
    oof2 = np.zeros_like(oof_meta)
    for fi, (tr_i, val_i) in enumerate(tscv_inner.split(X_all)):
        Xtr2, Xval2 = X_all.iloc[tr_i], X_all.iloc[val_i]
        ytr2        = y_all.iloc[tr_i]
        sc2  = StandardScaler()
        Xtr2_sc  = pd.DataFrame(sc2.fit_transform(Xtr2),  columns=Xtr2.columns)
        Xval2_sc = pd.DataFrame(sc2.transform(Xval2), columns=Xval2.columns)
        for mi, (name, model) in enumerate(base_models.items()):
            cs, ce = mi * 3, mi * 3 + 3
            try:
                if name == "MLP":
                    m = train_torch_mlp(Xtr2_sc, ytr2)
                    p = predict_torch_mlp(m, Xval2_sc)
                elif name == "XGBoost":
                    sw = compute_sample_weight("balanced", ytr2)
                    model.fit(Xtr2, ytr2, sample_weight=sw)
                    p  = model.predict_proba(Xval2)
                else:
                    model.fit(Xtr2, ytr2)
                    p = model.predict_proba(Xval2)
                oof2[val_i, cs:ce] = p
            except Exception:
                oof2[val_i, cs:ce] = 1/3

    oof_aug2     = np.hstack([oof2, X_all.values])
    mlp_meta_sc2 = StandardScaler()
    mlp_meta.fit(mlp_meta_sc2.fit_transform(oof_aug2), y_all)
    cat_meta.fit(oof_aug2, y_all)
    rf_meta.fit(oof_aug2, y_all)
    xgb_meta.fit(oof_aug2, y_all)

    state = {
        "base_models":    base_models,
        "model_names":    model_names,
        "sc_full":        sc_full,
        "torch_full":     torch_full_model,
        "cat_meta":       cat_meta,
        "mlp_meta":       mlp_meta,
        "mlp_meta_sc2":   mlp_meta_sc2,
        "rf_meta":        rf_meta,
        "xgb_meta":       xgb_meta,
        "hybrid_weights": hybrid_weights,
        "feats_train":    feats_train,
        "X_all":          X_all,
        "y_all":          y_all,
    }
    return state, None


# ─────────────────────────────────────────────────────────────────────────────
# Prediction for a single match
# ─────────────────────────────────────────────────────────────────────────────
def predict_match(state, feats_df, df_raw, home_team, away_team,
                  model_choice, market_h, market_d, market_a):
    """Build features for the requested matchup and run inference."""

    # Get most recent features for each team from feats_df
    home_rows = feats_df[feats_df["HomeTeam"] == home_team].sort_values("Date")
    away_rows = feats_df[feats_df["AwayTeam"] == away_team].sort_values("Date")
    home_as_away = feats_df[feats_df["AwayTeam"] == home_team].sort_values("Date")
    away_as_home = feats_df[feats_df["HomeTeam"] == away_team].sort_values("Date")

    # Combine to get most recent stats for each team
    all_home = pd.concat([home_rows, home_as_away]).sort_values("Date")
    all_away = pd.concat([away_rows, away_as_home]).sort_values("Date")

    def safe_last(df_t, col, default):
        sub = df_t[[col]].dropna()
        return float(sub.iloc[-1][col]) if len(sub) > 0 else default

    # Home team features (from their perspective as home)
    h_form       = safe_last(home_rows, "H_Form", 7)
    h_form10     = safe_last(home_rows, "H_Form10", 14)
    h_momentum   = safe_last(home_rows, "H_Momentum", 0)
    h_gd         = safe_last(home_rows, "H_GD", 0)
    h_attack     = safe_last(home_rows, "H_Attack", 1.2)
    h_defence    = safe_last(home_rows, "H_Defence", 1.1)
    h_score_std  = safe_last(home_rows, "H_Score_Std", 1.0)
    h_cs_rate    = safe_last(home_rows, "H_CS_Rate", 0.25)
    h_home_form  = safe_last(home_rows, "H_Home_Form", 7)
    h_rest       = 7
    h_recent     = 3
    elo_h        = safe_last(home_rows, "EloH", 1500)

    # Away team features
    a_form       = safe_last(away_rows, "A_Form", 7)
    a_form10     = safe_last(away_rows, "A_Form10", 14)
    a_momentum   = safe_last(away_rows, "A_Momentum", 0)
    a_gd         = safe_last(away_rows, "A_GD", 0)
    a_attack     = safe_last(away_rows, "A_Attack", 1.0)
    a_defence    = safe_last(away_rows, "A_Defence", 1.2)
    a_score_std  = safe_last(away_rows, "A_Score_Std", 1.0)
    a_cs_rate    = safe_last(away_rows, "A_CS_Rate", 0.20)
    a_away_form  = safe_last(away_rows, "A_Away_Form", 5)
    a_rest       = 7
    a_recent     = 3
    elo_a        = safe_last(away_rows, "EloA", 1500)

    elo_diff  = (elo_h + 75) - elo_a
    elo_close = 1 / (1 + abs(elo_diff) / 100)

    # H2H
    h2h_rows_hd = feats_df[(feats_df["HomeTeam"] == home_team) &
                            (feats_df["AwayTeam"] == away_team)].tail(5)
    h2h_win_rate  = (h2h_rows_hd["Target"] == 0).mean() if len(h2h_rows_hd) > 0 else 0.4
    h2h_draw_rate = (h2h_rows_hd["Target"] == 1).mean() if len(h2h_rows_hd) > 0 else 0.25

    # Form-based draw tendency
    h_res5_draw = safe_last(home_rows, "DrawTendency", 0.25)
    draw_tendency = h_res5_draw

    avg_goals = (h_attack + a_attack)

    # Market
    mh = market_h if market_h > 1.0 else 2.0
    md = market_d if market_d > 1.0 else 3.3
    ma = market_a if market_a > 1.0 else 3.5

    total_inv = 1/mh + 1/md + 1/ma
    imp_h = (1/mh) / total_inv
    imp_d = (1/md) / total_inv
    imp_a = (1/ma) / total_inv

    # League info
    div_code         = safe_last(home_rows, "Div_Code", 0)
    league_draw_rate = safe_last(home_rows, "League_Draw_Rate", 0.25)

    feat_vec = pd.DataFrame([{
        "EloDiff": elo_diff, "EloCloseness": elo_close,
        "H_Form": h_form, "A_Form": a_form,
        "H_Form10": h_form10, "A_Form10": a_form10,
        "H_Momentum": h_momentum, "A_Momentum": a_momentum,
        "H_GD": h_gd, "A_GD": a_gd,
        "H_Attack": h_attack, "A_Attack": a_attack,
        "H_Defence": h_defence, "A_Defence": a_defence,
        "H_Score_Std": h_score_std, "A_Score_Std": a_score_std,
        "AvgGoalsExpected": avg_goals,
        "H_CS_Rate": h_cs_rate, "A_CS_Rate": a_cs_rate,
        "DrawTendency": draw_tendency,
        "H_Home_Form": h_home_form, "A_Away_Form": a_away_form,
        "H_Rest": h_rest, "A_Rest": a_rest,
        "H_Recent_Games": h_recent, "A_Recent_Games": a_recent,
        "H2H_Win_Rate": h2h_win_rate, "H2H_Draw_Rate": h2h_draw_rate,
        "Market_H": mh, "Market_D": md, "Market_A": ma,
        "Imp_H": imp_h, "Imp_D": imp_d, "Imp_A": imp_a,
        "Imp_H_minus_A": imp_h - imp_a,
        "Imp_D_sq": imp_d ** 2,
        "Elo_x_ImpH": elo_diff * imp_h,
        "Div_Code": div_code,
        "League_Draw_Rate": league_draw_rate,
    }])

    sc_full = state["sc_full"]
    X_sc    = pd.DataFrame(sc_full.transform(feat_vec), columns=feat_vec.columns)
    base_models   = state["base_models"]
    model_names   = state["model_names"]
    torch_full    = state["torch_full"]
    hybrid_w      = state["hybrid_weights"]

    elo_probs = get_elo_win_probs_vec(
        np.array([elo_h]), np.array([elo_a]))[0]

    def get_base_prob(name):
        model = base_models[name]
        if name == "MLP":
            if torch_full is None:
                return np.array([1/3, 1/3, 1/3])
            return predict_torch_mlp(torch_full, X_sc)[0]
        return model.predict_proba(feat_vec)[0]

    # Stack features for meta-learners
    stk_probs = np.concatenate([get_base_prob(n) for n in model_names])
    stk_aug   = np.hstack([stk_probs, feat_vec.values[0]])

    def run_model(choice):
        if choice in base_models:
            return get_base_prob(choice)
        if choice == "MLP (PyTorch)":
            return get_base_prob("MLP")
        if choice == "Stacked (CatMeta)":
            return state["cat_meta"].predict_proba(stk_aug.reshape(1, -1))[0]
        if choice == "Stacked (MLPMeta)":
            sc2 = state["mlp_meta_sc2"]
            return state["mlp_meta"].predict_proba(sc2.transform(stk_aug.reshape(1, -1)))[0]
        if choice == "Stacked (RFMeta)":
            return state["rf_meta"].predict_proba(stk_aug.reshape(1, -1))[0]
        if choice == "Stacked (XGBMeta)":
            return state["xgb_meta"].predict_proba(stk_aug.reshape(1, -1))[0]
        if choice.startswith("Hybrid"):
            # extract base model name from "Hybrid (XGBoost-Elo)"
            inner = choice[len("Hybrid ("):-len("-Elo)")]
            bp    = get_base_prob(inner)
            hw    = hybrid_w.get(inner, {"model_w": 0.7, "elo_w": 0.3})
            return hw["model_w"] * bp + hw["elo_w"] * elo_probs
        if choice == "Best Auto-Selected":
            # Simple ensemble of all base models
            all_p = np.mean([get_base_prob(n) for n in model_names if n != "MLP" or torch_full is not None], axis=0)
            return all_p
        return np.array([1/3, 1/3, 1/3])

    probs = run_model(model_choice)
    # Ensure sums to 1
    probs = np.clip(probs, 0, 1)
    probs = probs / probs.sum()
    return probs, elo_probs


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("⚽ Multi-League Match Predictor")
st.caption("Upload your historical match data, train, and predict any fixture.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")
    uploaded = st.file_uploader("Upload `all_leagues.csv`", type=["csv"])

    scenario_name = st.selectbox("League Scenario", list(SCENARIOS.keys()), index=0)
    model_choice  = st.selectbox("Model", MODEL_OPTIONS, index=3)

    st.markdown("---")
    st.markdown("**📊 Betting Odds** *(optional — improves accuracy)*")
    market_h = st.number_input("Home Win Odds", min_value=1.01, value=2.00, step=0.05, format="%.2f")
    market_d = st.number_input("Draw Odds",     min_value=1.01, value=3.30, step=0.05, format="%.2f")
    market_a = st.number_input("Away Win Odds", min_value=1.01, value=3.50, step=0.05, format="%.2f")

    st.markdown("---")
    gpu_status = "🚀 GPU active" if HAS_GPU else "💻 CPU mode"
    st.caption(gpu_status)

# ── Main panel ────────────────────────────────────────────────────────────────
if uploaded is None:
    st.info("👈 Upload your `all_leagues.csv` file in the sidebar to get started.")
    st.markdown("""
    **Expected CSV columns:** `Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`, `Div`,
    and optional odds columns like `B365H`, `B365D`, `B365A`.
    """)
    st.stop()

# Load & engineer features
file_bytes = uploaded.read()
file_hash  = str(hash(file_bytes[:5000]))  # cheap hash for cache key

with st.spinner("Loading data…"):
    df_raw, feats_df, elo_map, team_hist_map = load_and_engineer(file_bytes)

st.success(f"✅ Loaded {len(df_raw):,} matches  |  "
           f"{feats_df['HomeTeam'].nunique()} teams  |  "
           f"{df_raw['Date'].min().year}–{df_raw['Date'].max().year}")

# Team selection
all_teams = sorted(set(feats_df["HomeTeam"]) | set(feats_df["AwayTeam"]))

col1, col2 = st.columns(2)
with col1:
    st.subheader("🏠 Home Team")
    home_team = st.selectbox("Select Home Team", all_teams, key="home")
with col2:
    st.subheader("✈️ Away Team")
    away_candidates = [t for t in all_teams if t != home_team]
    away_team = st.selectbox("Select Away Team", away_candidates, key="away")

st.markdown("---")

# Train button
if st.button("🚀 Train & Predict", use_container_width=True):
    if home_team == away_team:
        st.error("Home and Away teams must be different!")
        st.stop()

    # Serialise feats_df for cache key (use first 5k rows as proxy)
    feats_sample = feats_df.head(5000).to_json()
    df_sample    = df_raw.head(5000).to_json()

    with st.spinner(f"Training models for **{scenario_name}**…"):
        state, err = train_models(file_hash, scenario_name,
                                   str(SCENARIOS[scenario_name]),
                                   feats_sample, df_sample)

    if err:
        st.error(f"Training failed: {err}")
        st.stop()

    st.success("✅ Models trained!")

    # Predict
    probs, elo_probs = predict_match(
        state, feats_df, df_raw,
        home_team, away_team, model_choice,
        market_h, market_d, market_a
    )
    p_home, p_draw, p_away = float(probs[0]), float(probs[1]), float(probs[2])
    outcome_idx = int(np.argmax(probs))
    outcome_map = {0: f"🏠 {home_team} Win", 1: "🤝 Draw", 2: f"✈️ {away_team} Win"}
    predicted   = outcome_map[outcome_idx]
    conf_colour = "#1db954" if probs[outcome_idx] >= 0.50 else ("#f0ad00" if probs[outcome_idx] >= 0.38 else "#e55353")

    st.markdown("---")
    st.markdown(f"### 🔮 Prediction: **{home_team}** vs **{away_team}**")
    st.caption(f"Model: `{model_choice}`  |  Scenario: `{scenario_name}`")

    # Predicted outcome banner
    st.markdown(f"""
    <div class="winner-card">
        <div style="font-size:1rem;color:#aaa;margin-bottom:0.3rem;">Predicted Outcome</div>
        <div style="font-size:2.4rem;font-weight:900;color:{conf_colour};">{predicted}</div>
        <div style="font-size:1.1rem;color:#ccc;margin-top:0.3rem;">
            Confidence: <b style="color:{conf_colour};">{probs[outcome_idx]*100:.1f}%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Probability cards
    c1, c2, c3 = st.columns(3)
    for col, label, prob, colour in [
        (c1, f"🏠 {home_team} Win", p_home, "#4a9eff"),
        (c2, "🤝 Draw",             p_draw, "#f0ad00"),
        (c3, f"✈️ {away_team} Win", p_away, "#e55353"),
    ]:
        with col:
            st.markdown(f"""
            <div class="pred-card">
                <div class="pred-title">{label}</div>
                <div class="pred-value" style="color:{colour};">{prob*100:.1f}%</div>
                <div class="pred-pct">Implied odds: {1/prob:.2f}x</div>
            </div>
            """, unsafe_allow_html=True)

    # Elo baseline comparison
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📐 Elo Baseline Probabilities"):
        e1, e2, e3 = st.columns(3)
        for col, label, prob in [
            (e1, f"Home Win", float(elo_probs[0])),
            (e2, "Draw",      float(elo_probs[1])),
            (e3, "Away Win",  float(elo_probs[2])),
        ]:
            with col:
                st.metric(label, f"{prob*100:.1f}%")

    # Value bets
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("💰 Value Bet Analysis"):
        labels  = [f"{home_team} Win", "Draw", f"{away_team} Win"]
        markets = [market_h, market_d, market_a]
        prob_list = [p_home, p_draw, p_away]
        rows = []
        for lbl, odds, p in zip(labels, markets, prob_list):
            implied = 1 / odds
            edge    = (p - implied) * 100
            rows.append({
                "Outcome": lbl,
                "Model Prob": f"{p*100:.1f}%",
                "Market Odds": f"{odds:.2f}",
                "Implied Prob": f"{implied*100:.1f}%",
                "Edge": f"+{edge:.1f}%" if edge > 0 else f"{edge:.1f}%",
                "Value?": "✅ Yes" if edge > 3 else ("⚠️ Marginal" if edge > 0 else "❌ No"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

else:
    # Show team info while waiting
    st.info("👆 Press **Train & Predict** to run the model.")

    if home_team and away_team and home_team != away_team:
        col1, col2 = st.columns(2)
        for col, team, label in [(col1, home_team, "Home"), (col2, away_team, "Away")]:
            with col:
                team_rows = feats_df[feats_df["HomeTeam"] == team].tail(5)
                st.markdown(f"**Recent {label} form — {team}**")
                if len(team_rows) > 0:
                    last = team_rows.iloc[-1]
                    st.metric("Elo Rating", f"{last['EloH']:.0f}")
                    st.metric("Last 5 Form Pts", f"{last['H_Form']:.0f} / 15")
                    st.metric("Attack Avg (10g)", f"{last['H_Attack']:.2f}")
                else:
                    st.caption("No data found for this team.")