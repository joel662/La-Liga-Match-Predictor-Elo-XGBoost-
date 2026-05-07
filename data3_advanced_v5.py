"""
data3_advanced_v5.py — Unified Multi-League Match Predictor
============================================================
Updated from v4 to address CCIDSA 2026 reviewer feedback:

  NEW in v5
  ---------
  • get_stats() now returns: accuracy, home recall, draw recall, away recall,
    macro-F1, balanced accuracy, AND soft-probability log loss (replaces one-hot LL)
  • All result tables include the full metric set
  • Elo weight summary CSV saved per scenario (elo_weights_summary.csv)
  • Ablation study: Odds-only / Elo-only / Form-only / Odds+Form / Full ML / Full+Elo
    run inside each pipeline call; saved to ablation_results.csv
  • Rolling-origin backtest over three historical seasons (2022/23, 2023/24, 2024/25)
    run for top-5 individual leagues using the FULL 21-variant pipeline per season;
    saved to rolling_backtest_results.csv

Structure
---------
  SECTION 0 : Imports & GPU detection
  SECTION 1 : Data loading & odds imputation
  SECTION 2 : Feature engineering (run ONCE over all matches)
  SECTION 3 : PyTorch MLP definition & helpers
  SECTION 4 : Core pipeline function  run_pipeline(scenario)
  SECTION 5 : Ablation helper
  SECTION 6 : Rolling-origin backtest (full 21-variant pipeline per season)
  SECTION 7 : Scenario definitions & execution
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — Imports & GPU Detection
# ─────────────────────────────────────────────────────────────────────────────
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

import os
import json
import pickle
import warnings
from collections import deque

import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── GPU probing ───────────────────────────────────────────────────────────────
print("🔍 Checking for GPU acceleration...")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HAS_GPU  = DEVICE.type == "cuda"
NUM_CPUS = os.cpu_count() or 4

if HAS_GPU:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🚀 GPU: {gpu_name}  ({gpu_mem:.1f} GB VRAM)  |  "
          f"CUDA {torch.version.cuda}  |  PyTorch {torch.__version__}")
else:
    print(f"💻 No CUDA GPU — running on CPU ({NUM_CPUS} cores)")


def _probe(fn):
    try:
        fn(); return True
    except Exception:
        return False


XGB_GPU = HAS_GPU and _probe(lambda: XGBClassifier(
    device="cuda", tree_method="hist",
    objective="multi:softprob", num_class=3,
).fit(np.zeros((4, 2)), np.array([0, 1, 2, 0])))

LGB_GPU = HAS_GPU and _probe(lambda: LGBMClassifier(
    device="gpu", n_estimators=1, verbose=-1,
).fit(np.zeros((4, 2)), np.array([0, 1, 2, 0])))

CAT_GPU = HAS_GPU and _probe(lambda: CatBoostClassifier(
    task_type="GPU", devices="0", iterations=1, verbose=False,
).fit(np.zeros((4, 2)), np.array([0, 1, 2, 0])))

print(f"  XGBoost  GPU: {'✅' if XGB_GPU else '⚠️  CPU fallback'}")
print(f"  LightGBM GPU: {'✅' if LGB_GPU else '⚠️  CPU fallback'}")
print(f"  CatBoost GPU: {'✅' if CAT_GPU else '⚠️  CPU fallback'}")
print(f"  PyTorch  GPU: {'✅' if HAS_GPU  else '⚠️  CPU fallback'}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Data Loading & Odds Imputation
# ─────────────────────────────────────────────────────────────────────────────
print("\n📂 Loading dataset...")
df = pd.read_csv("all_leagues.csv")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
print(f"📊 Dataset: {len(df):,} matches  "
      f"({df['Date'].min().year}–{df['Date'].max().year})")

# Vectorised odds imputation
for suffix in ["H", "D", "A"]:
    for prefix in ["PS", "BbAv"]:
        col, ref = prefix + suffix, "B365" + suffix
        if col in df.columns and ref in df.columns:
            df[col] = df[col].fillna(df[ref])

for suffix in ["H", "D", "A"]:
    cols = [c + suffix for c in ["B365", "BbAv", "PS"] if c + suffix in df.columns]
    df[f"Market_{suffix}"] = df[cols].mean(axis=1)

print("✅ Odds imputation complete.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Feature Engineering  (runs once over the full dataset)
# ─────────────────────────────────────────────────────────────────────────────
def expected_result(elo_a: float, elo_b: float) -> float:
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def update_elo_mov(elo_home, elo_away, result, goal_diff, k=20, home_adv=75):
    """Margin-of-victory scaled Elo update."""
    exp_home   = expected_result(elo_home + home_adv, elo_away)
    score_home = 1 if result == 1 else (0.5 if result == 0 else 0)
    mov        = min(np.log(abs(goal_diff) + 1) + 1.0, 2.5)
    k_eff      = k * mov
    return (
        elo_home + k_eff * (score_home - exp_home),
        elo_away + k_eff * ((1 - score_home) - (1 - exp_home)),
    )


def get_elo_win_probs_vec(elo_h, elo_a, home_adv=75, draw_prob=0.25):
    """Vectorised Elo win-probability helper → (N, 3) array [home, draw, away]."""
    win_h = 1 / (1 + 10 ** ((elo_a - (elo_h + home_adv)) / 400))
    win_a = 1 / (1 + 10 ** (((elo_h + home_adv) - elo_a) / 400))
    rem   = 1.0 - draw_prob
    p_d   = np.full_like(win_h, draw_prob, dtype=float)
    return np.column_stack([win_h * rem, p_d, win_a * rem])


HIST_LEN = 10
PTS_MAP  = {"H": 3, "D": 1, "A": 0}

teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
elo   = {t: 1500.0 for t in teams}


def _make_hist():
    return {k: deque(maxlen=HIST_LEN) for k in
            ("dates", "results", "goals_for", "goals_against",
             "home_results", "away_results", "clean_sheets")}


team_hist = {t: _make_hist() for t in teams}
h2h: dict = {}


def get_pts(dq, n=5):
    return sum(PTS_MAP.get(r, 0) for r in list(dq)[-n:])


feature_rows = []
print(f"⚙️  Building features over {len(df):,} matches (this takes ~30 s)…")

for _, row in df.iterrows():
    home, away, date = row["HomeTeam"], row["AwayTeam"], row["Date"]
    hg, ag    = int(row["FTHG"]), int(row["FTAG"])
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

    h_attack   = np.mean(hgf[-10:]) if hgf else 1.2
    h_defence  = np.mean(hga[-10:]) if hga else 1.1
    a_attack   = np.mean(agf[-10:]) if agf else 1.0
    a_defence  = np.mean(aga[-10:]) if aga else 1.2

    h_score_std = np.std(hgf[-10:]) if len(hgf) >= 3 else 1.0
    a_score_std = np.std(agf[-10:]) if len(agf) >= 3 else 1.0
    h_cs_rate   = np.mean(list(hh["clean_sheets"])[-10:]) if hh["clean_sheets"] else 0.25
    a_cs_rate   = np.mean(list(ah["clean_sheets"])[-10:]) if ah["clean_sheets"] else 0.20

    h_dates, a_dates = list(hh["dates"]), list(ah["dates"])
    h_rest   = int((date - h_dates[-1]) / pd.Timedelta(days=1)) if h_dates else 14
    a_rest   = int((date - a_dates[-1]) / pd.Timedelta(days=1)) if a_dates else 14
    h_recent = sum(1 for d in h_dates[-8:]
                   if int((date - d) / pd.Timedelta(days=1)) <= 14)
    a_recent = sum(1 for d in a_dates[-8:]
                   if int((date - d) / pd.Timedelta(days=1)) <= 14)

    h2h_hist      = h2h.get((home, away), [])[-5:]
    h2h_win_rate  = h2h_hist.count("H") / max(len(h2h_hist), 1)
    h2h_draw_rate = h2h_hist.count("D") / max(len(h2h_hist), 1)

    h_home_form = get_pts(hh["home_results"], 5)
    a_away_form = get_pts(ah["away_results"], 5)

    h_form3    = get_pts(hh["results"], 3)
    a_form3    = get_pts(ah["results"], 3)
    h_momentum = h_form3 / 9 - (h_form10 / 30 if h_form10 else 0)
    a_momentum = a_form3 / 9 - (a_form10 / 30 if a_form10 else 0)

    feature_rows.append({
        "Date": date, "HomeTeam": home, "AwayTeam": away,
        "EloH": elo_h, "EloA": elo_a,
        "EloDiff": elo_diff, "EloCloseness": elo_close,
        "H_Form": h_form,    "A_Form": a_form,
        "H_Form10": h_form10, "A_Form10": a_form10,
        "H_Momentum": h_momentum, "A_Momentum": a_momentum,
        "H_GD": h_gd, "A_GD": a_gd,
        "H_Attack": h_attack, "A_Attack": a_attack,
        "H_Defence": h_defence, "A_Defence": a_defence,
        "H_Score_Std": h_score_std, "A_Score_Std": a_score_std,
        "AvgGoalsExpected": (np.mean(h_gf5) if h_gf5 else 1.2)
                           + (np.mean(a_gf5) if a_gf5 else 1.0),
        "H_CS_Rate": h_cs_rate, "A_CS_Rate": a_cs_rate,
        "DrawTendency": draw_tendency,
        "H_Home_Form": h_home_form, "A_Away_Form": a_away_form,
        "H_Rest": min(h_rest, 20), "A_Rest": min(a_rest, 20),
        "H_Recent_Games": h_recent, "A_Recent_Games": a_recent,
        "H2H_Win_Rate": h2h_win_rate, "H2H_Draw_Rate": h2h_draw_rate,
        "Market_H": row["Market_H"], "Market_D": row["Market_D"],
        "Market_A": row["Market_A"],
        "Target": 0 if res == 1 else (1 if res == 0 else 2),
    })

    res_str = "H" if res == 1 else ("D" if res == 0 else "A")
    for t, gf, ga, is_home in [(home, hg, ag, True), (away, ag, hg, False)]:
        pov = res_str if is_home else (
              "A" if res_str == "H" else ("H" if res_str == "A" else "D"))
        hst = team_hist[t]
        hst["dates"].append(date)
        hst["results"].append(pov)
        hst["goals_for"].append(gf)
        hst["goals_against"].append(ga)
        hst["clean_sheets"].append(1 if ga == 0 else 0)
        (hst["home_results"] if is_home else hst["away_results"]).append(pov)
    h2h.setdefault((home, away), []).append(res_str)
    elo[home], elo[away] = update_elo_mov(elo_h, elo_a, res, goal_diff)

print("✅ Feature loop complete.")

# Post-loop enrichment ────────────────────────────────────────────────────────
feats_df = pd.DataFrame(feature_rows)

df["Div_Code"]         = df["Div"].astype("category").cat.codes
league_draw_rates      = df.groupby("Div")["FTR"].apply(lambda x: (x == "D").mean())
df["League_Draw_Rate"] = df["Div"].map(league_draw_rates)
feats_df["Div_Code"]         = df["Div_Code"].values
feats_df["League_Draw_Rate"] = df["League_Draw_Rate"].values

total_inv = ((1 / feats_df["Market_H"])
             + (1 / feats_df["Market_D"])
             + (1 / feats_df["Market_A"]))
feats_df["Imp_H"]         = (1 / feats_df["Market_H"]) / total_inv
feats_df["Imp_D"]         = (1 / feats_df["Market_D"]) / total_inv
feats_df["Imp_A"]         = (1 / feats_df["Market_A"]) / total_inv
feats_df["Imp_H_minus_A"] = feats_df["Imp_H"] - feats_df["Imp_A"]
feats_df["Imp_D_sq"]      = feats_df["Imp_D"] ** 2
feats_df["Elo_x_ImpH"]   = feats_df["EloDiff"] * feats_df["Imp_H"]

# ── Feature column groups used by ablation study ──────────────────────────────
ODDS_COLS = [
    "Market_H", "Market_D", "Market_A",
    "Imp_H", "Imp_D", "Imp_A", "Imp_H_minus_A", "Imp_D_sq",
]
ELO_COLS = ["EloDiff", "EloCloseness", "Elo_x_ImpH"]
FORM_COLS = [
    "H_Form", "A_Form", "H_Form10", "A_Form10",
    "H_Momentum", "A_Momentum", "H_GD", "A_GD",
    "H_Attack", "A_Attack", "H_Defence", "A_Defence",
    "H_Score_Std", "A_Score_Std", "H_CS_Rate", "A_CS_Rate",
    "DrawTendency", "H_Home_Form", "A_Away_Form",
    "H_Rest", "A_Rest", "H_Recent_Games", "A_Recent_Games",
    "H2H_Win_Rate", "H2H_Draw_Rate", "AvgGoalsExpected",
]
META_COLS = ["Div_Code", "League_Draw_Rate"]

FEATURE_COLS = ELO_COLS + FORM_COLS + ODDS_COLS + META_COLS

nan_count = feats_df[FEATURE_COLS].isna().sum().sum()
if nan_count:
    print(f"⚠️  Imputing {nan_count} NaN values with column means.")
    feats_df[FEATURE_COLS] = (feats_df[FEATURE_COLS]
                               .fillna(feats_df[FEATURE_COLS].mean())
                               .fillna(0))

print(f"✅ Feature engineering done — {len(FEATURE_COLS)} features, "
      f"{len(feats_df):,} rows.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — PyTorch MLP (AMP + OneCycleLR + class-weighted loss)
# ─────────────────────────────────────────────────────────────────────────────
class TorchMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(256, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),
            nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.net(x)


def train_torch_mlp(X_train: pd.DataFrame, y_train: pd.Series,
                    epochs: int = 40, batch_size: int = 1024):
    X_np = X_train.values.astype(np.float32)
    y_np = y_train.values.astype(np.int64)
    counts = np.bincount(y_np, minlength=3)
    cw     = torch.tensor(len(y_np) / (3 * counts + 1e-8),
                          dtype=torch.float32).to(DEVICE)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_np), torch.from_numpy(y_np)),
        batch_size=batch_size, shuffle=True,
        pin_memory=HAS_GPU, num_workers=0,
    )
    model     = TorchMLP(X_train.shape[1]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-3,
        steps_per_epoch=len(loader), epochs=epochs)
    scaler_amp = GradScaler(enabled=HAS_GPU)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx = bx.to(DEVICE, non_blocking=True)
            by = by.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=HAS_GPU):
                loss = criterion(model(bx), by)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            scheduler.step()
    return model


def predict_torch_mlp(model: TorchMLP, X) -> np.ndarray:
    model.eval()
    X_np = (X.values.astype(np.float32)
            if hasattr(X, "values") else np.asarray(X, dtype=np.float32))
    with torch.no_grad(), autocast(enabled=HAS_GPU):
        t = torch.from_numpy(X_np).to(DEVICE, non_blocking=True)
        return torch.softmax(model(t), dim=1).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Core Pipeline Function
# ─────────────────────────────────────────────────────────────────────────────
CUTOFF   = pd.Timestamp("2025-08-01")
TSCV     = TimeSeriesSplit(n_splits=3)
TOP5_DIV = ["E0", "SP1", "I1", "D1", "F1"]


def get_stats(preds: np.ndarray, y_true: np.ndarray,
              probs: np.ndarray | None = None) -> dict:
    """
    Compute the full metric set required by revised paper tables.

    Parameters
    ----------
    preds   : (N,) integer class predictions  {0=H, 1=D, 2=A}
    y_true  : (N,) integer ground-truth labels
    probs   : (N, 3) soft probability matrix — used for log loss.
              Falls back to one-hot if None (legacy behaviour).

    Returns
    -------
    dict with keys:
        correct, pct_correct,
        home_recall, draw_recall, away_recall,
        macro_f1, balanced_acc,
        log_loss_soft   ← average per-match log loss using soft probs
    """
    correct = int((preds == y_true).sum())
    pct     = round(100 * correct / len(y_true), 2)

    recalls = recall_score(y_true, preds, labels=[0, 1, 2],
                           average=None, zero_division=0)
    home_r = round(100 * float(recalls[0]), 1)
    draw_r = round(100 * float(recalls[1]), 1)
    away_r = round(100 * float(recalls[2]), 1)

    macro_f1  = round(float(f1_score(y_true, preds,
                                     average="macro", zero_division=0)), 4)
    bal_acc   = round(float(balanced_accuracy_score(y_true, preds)), 4)

    if probs is not None:
        ll = round(float(log_loss(y_true, probs)), 4)
    else:
        ll = round(float(log_loss(y_true, np.eye(3)[preds])), 4)

    return {
        "correct":       correct,
        "pct_correct":   pct,
        "home_recall":   home_r,
        "draw_recall":   draw_r,
        "away_recall":   away_r,
        "macro_f1":      macro_f1,
        "balanced_acc":  bal_acc,
        "log_loss_soft": ll,
    }


def _row(model_name: str, preds: np.ndarray, y_true: np.ndarray,
         probs: np.ndarray | None = None) -> dict:
    """Convenience wrapper that prepends the model name to get_stats output."""
    s = get_stats(preds, y_true, probs)
    return {"Model": model_name, **s}


def run_pipeline(scenario_name: str, div_filter: list | None,
                 train_cutoff: pd.Timestamp | None = None,
                 test_cutoff:  pd.Timestamp | None = None,
                 save_models: bool = True):
    """
    Run the full ML pipeline for a given scenario.

    Parameters
    ----------
    scenario_name  : Human-readable label (e.g. "Premier League")
    div_filter     : List of Div codes to keep, or None for all leagues.
    train_cutoff   : Upper bound for training data (default: CUTOFF).
    test_cutoff    : Upper bound for test data (default: no upper bound).
    save_models    : Whether to pickle models and write CSVs to disk.
    """
    tc = train_cutoff if train_cutoff is not None else CUTOFF

    print(f"\n{'='*65}")
    print(f"  🏆  PIPELINE: {scenario_name}  "
          f"(train < {tc.date()}, test >= {tc.date()})")
    print(f"{'='*65}")

    # ── 4a. Filter data ───────────────────────────────────────────────────────
    if div_filter:
        mask      = df["Div"].isin(div_filter)
        cur_feats = feats_df[mask.values].copy()
    else:
        cur_feats = feats_df.copy()

    feats_train = cur_feats[cur_feats["Date"] < tc].copy()
    feats_test_ = cur_feats[cur_feats["Date"] >= tc].copy()
    if test_cutoff is not None:
        feats_test_ = feats_test_[feats_test_["Date"] < test_cutoff].copy()
    feats_test = feats_test_

    print(f"  📊 Total: {len(cur_feats):,}  |  "
          f"Train: {len(feats_train):,}  |  "
          f"Holdout: {len(feats_test):,}")

    if len(feats_train) < 500:
        print("  ⚠️  Too few training rows — skipping scenario.")
        return None

    X_all = feats_train[FEATURE_COLS]
    y_all = feats_train["Target"]

    # ── 4b. Optuna hyperparameter tuning (50 trials each) ────────────────────
    tscv_tune = TimeSeriesSplit(n_splits=3)

    def cv_score(model_fn, X, y, sw_fn=None):
        scores = []
        for tr, val in tscv_tune.split(X):
            m = model_fn()
            if sw_fn:
                m.fit(X.iloc[tr], y.iloc[tr],
                      sample_weight=sw_fn(y.iloc[tr]))
            else:
                m.fit(X.iloc[tr], y.iloc[tr])
            scores.append(
                balanced_accuracy_score(y.iloc[val], m.predict(X.iloc[val])))
        return float(np.mean(scores))

    print("\n  🔬 Optuna tuning — XGBoost…")

    def xgb_obj(trial):
        p = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 3, 7),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2,
                                              log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            gamma=trial.suggest_float("gamma", 0.0, 1.0),
            device="cuda" if XGB_GPU else "cpu",
            tree_method="hist",
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", verbosity=0,
        )
        return cv_score(lambda: XGBClassifier(**p), X_all, y_all,
                        sw_fn=lambda y: compute_sample_weight("balanced", y))

    xgb_study = optuna.create_study(direction="maximize")
    xgb_study.optimize(xgb_obj, n_trials=50, show_progress_bar=True,
                       n_jobs=1)
    best_xgb = {**xgb_study.best_params,
                "objective": "multi:softprob", "num_class": 3,
                "eval_metric": "mlogloss", "verbosity": 0,
                "tree_method": "hist",
                "device": "cuda" if XGB_GPU else "cpu"}
    print(f"  ✅ XGBoost best bal-acc: {xgb_study.best_value:.4f}")

    print("  🔬 Optuna tuning — LightGBM…")

    def lgb_obj(trial):
        p = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2,
                                              log=True),
            num_leaves=trial.suggest_int("num_leaves", 20, 100),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            device="gpu" if LGB_GPU else "cpu",
            n_jobs=1 if LGB_GPU else -1,
            class_weight="balanced", random_state=42, verbose=-1,
        )
        return cv_score(lambda: LGBMClassifier(**p), X_all, y_all)

    lgb_study = optuna.create_study(direction="maximize")
    lgb_study.optimize(lgb_obj, n_trials=50, show_progress_bar=True,
                       n_jobs=1)
    best_lgb = {**lgb_study.best_params,
                "class_weight": "balanced", "random_state": 42,
                "verbose": -1,
                "device": "gpu" if LGB_GPU else "cpu",
                "n_jobs": 1 if LGB_GPU else -1}
    print(f"  ✅ LightGBM best bal-acc: {lgb_study.best_value:.4f}")

    print("  🔬 Optuna tuning — CatBoost…")

    def cat_obj(trial):
        p = dict(
            iterations=trial.suggest_int("iterations", 100, 500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2,
                                              log=True),
            depth=trial.suggest_int("depth", 4, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-3, 10,
                                            log=True),
            task_type="GPU" if CAT_GPU else "CPU",
            **({"devices": "0"} if CAT_GPU else {}),
            class_weights=[1.0, 1.5, 1.0],
            verbose=False, random_state=42,
        )
        return cv_score(lambda: CatBoostClassifier(**p), X_all, y_all)

    cat_study = optuna.create_study(direction="maximize")
    cat_study.optimize(cat_obj, n_trials=50, show_progress_bar=True,
                       n_jobs=1)
    best_cat = {**cat_study.best_params,
                "verbose": False, "random_state": 42,
                "class_weights": [1.0, 1.5, 1.0],
                "task_type": "GPU" if CAT_GPU else "CPU"}
    if CAT_GPU:
        best_cat["devices"] = "0"
    print(f"  ✅ CatBoost best bal-acc: {cat_study.best_value:.4f}")

    # ── 4c. Base-model definitions ────────────────────────────────────────────
    base_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", C=0.5, n_jobs=-1),
        "Random Forest":       RandomForestClassifier(
            n_estimators=400, max_depth=12, class_weight="balanced",
            min_samples_leaf=3, random_state=42, n_jobs=-1),
        "Linear SVM":          CalibratedClassifierCV(
            LinearSVC(C=0.5, class_weight="balanced", max_iter=2000)),
        "XGBoost":             XGBClassifier(**best_xgb),
        "LightGBM":            LGBMClassifier(**best_lgb),
        "CatBoost":            CatBoostClassifier(**best_cat),
        "MLP":                 None,   # sentinel — handled via TorchMLP
        "HistGBM":             HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.05,
            class_weight="balanced", random_state=42),
    }
    model_names = list(base_models.keys())
    n_models    = len(model_names)

    # ── 4d. OOF generation ────────────────────────────────────────────────────
    oof_meta = np.zeros((len(X_all), n_models * 3))
    print("\n  🔄 Generating OOF predictions (3-fold time-series CV)…")

    for fold_i, (tr_idx, val_idx) in enumerate(TSCV.split(X_all)):
        print(f"    Fold {fold_i + 1}/3  "
              f"(train={len(tr_idx):,}  val={len(val_idx):,})")
        Xtr,  Xval = X_all.iloc[tr_idx], X_all.iloc[val_idx]
        ytr,  yval = y_all.iloc[tr_idx], y_all.iloc[val_idx]

        sc      = StandardScaler()
        Xtr_sc  = pd.DataFrame(sc.fit_transform(Xtr),  columns=Xtr.columns)
        Xval_sc = pd.DataFrame(sc.transform(Xval),     columns=Xval.columns)

        for m_idx, (name, model) in enumerate(
                tqdm(base_models.items(),
                     desc=f"    fold {fold_i+1}", leave=False)):
            cs, ce = m_idx * 3, m_idx * 3 + 3
            if name == "MLP":
                m = train_torch_mlp(Xtr_sc, ytr)
                p = predict_torch_mlp(m, Xval_sc)
            elif name == "XGBoost":
                sw = compute_sample_weight("balanced", ytr)
                model.fit(Xtr, ytr, sample_weight=sw)
                p  = model.predict_proba(Xval)
            else:
                model.fit(Xtr, ytr)
                p = model.predict_proba(Xval)
            oof_meta[val_idx, cs:ce] = p

    # ── 4e. Calibration on last fold ─────────────────────────────────────────
    tr_last, val_last = list(TSCV.split(X_all))[-1]
    X_train, X_test   = X_all.iloc[tr_last], X_all.iloc[val_last]
    y_train, y_test   = y_all.iloc[tr_last], y_all.iloc[val_last]
    y_test_vals       = y_test.values

    sc_cal  = StandardScaler()
    Xtr_sc  = pd.DataFrame(sc_cal.fit_transform(X_train),
                            columns=X_train.columns)
    Xte_sc  = pd.DataFrame(sc_cal.transform(X_test),
                            columns=X_test.columns)

    print("\n  📦 Calibrating base models (isotonic)…")
    ml_probs:   dict[str, np.ndarray] = {}
    cal_models: dict = {}
    torch_cal_model = None

    for name, model in base_models.items():
        if name == "MLP":
            torch_cal_model = train_torch_mlp(Xtr_sc, y_train)
            ml_probs[name]  = predict_torch_mlp(torch_cal_model, Xte_sc)
            continue
        if name == "XGBoost":
            sw = compute_sample_weight("balanced", y_train)
            model.fit(X_train, y_train, sample_weight=sw)
        else:
            model.fit(X_train, y_train)
        cal = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
        cal.fit(X_test, y_test_vals)
        ml_probs[name]   = cal.predict_proba(X_test)
        cal_models[name] = cal

    # ── 4f. Stacked meta-learners ─────────────────────────────────────────────
    print("  🔮 Training 4 stacked meta-learners…")
    oof_aug = np.hstack([oof_meta, X_all.values])

    cat_meta = CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=6,
        verbose=False, random_state=42,
        task_type="GPU" if CAT_GPU else "CPU",
        **({"devices": "0"} if CAT_GPU else {}),
    )
    cat_meta.fit(oof_aug, y_all)

    mlp_meta_sc = StandardScaler()
    mlp_meta    = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64), activation="relu",
        max_iter=400, early_stopping=True, random_state=42)
    mlp_meta.fit(mlp_meta_sc.fit_transform(oof_aug), y_all)

    rf_meta = RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    rf_meta.fit(oof_aug, y_all)

    xgb_meta = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        eval_metric="mlogloss", verbosity=0, tree_method="hist",
        device="cuda" if XGB_GPU else "cpu", random_state=42)
    xgb_meta.fit(oof_aug, y_all)

    # ── 4g. Hybrid weight optimisation (vectorised, macro-F1) ────────────────
    elo_probs_val = get_elo_win_probs_vec(
        feats_train.iloc[val_last]["EloH"].values,
        feats_train.iloc[val_last]["EloA"].values,
    )
    weights_grid  = np.round(np.arange(0.0, 1.01, 0.01), 2)

    hybrid_weights: dict[str, dict] = {}

    print("  🔗 Optimising hybrid weights (Macro-F1, vectorised)…")
    for name in model_names:
        if name not in ml_probs:
            continue
        base_p  = ml_probs[name]
        blended = (weights_grid[:, None, None] * base_p[None]
                   + (1 - weights_grid)[:, None, None]
                   * elo_probs_val[None])
        f1s     = np.array([
            f1_score(y_test_vals, np.argmax(b, axis=1), average="macro")
            for b in blended
        ])
        best_w  = float(weights_grid[np.argmax(f1s)])
        hybrid_weights[name] = {
            "model_w": best_w,
            "elo_w":   round(1 - best_w, 2),
        }
        print(f"    ⭐ Hybrid ({name}-Elo): "
              f"model_w={best_w:.2f}  elo_w={1-best_w:.2f}  "
              f"macro-F1={f1s.max():.4f}")

    # ── 4h. Draw-threshold tuning ─────────────────────────────────────────────
    best_ba  = max(ml_probs,
                   key=lambda n: balanced_accuracy_score(
                       y_test_vals, np.argmax(ml_probs[n], axis=1)))
    best_bp  = ml_probs[best_ba]
    thresh_arr  = np.arange(0.25, 0.50, 0.01)
    thresh_accs = [
        accuracy_score(
            y_test_vals,
            np.where(best_bp[:, 1] >= t, 1,
                     np.where(best_bp[:, 0] > best_bp[:, 2], 0, 2)))
        for t in thresh_arr
    ]
    best_thresh = float(thresh_arr[np.argmax(thresh_accs)])
    print(f"\n  🎯 Optimal draw threshold ({best_ba}): {best_thresh:.2f}")

    # ── 4i. Full retrain on all training data ─────────────────────────────────
    print("\n  🔄 Retraining all models on full training data…")
    sc_full  = StandardScaler()
    X_all_sc = pd.DataFrame(sc_full.fit_transform(X_all),
                             columns=X_all.columns)
    torch_full_model = None

    for name, model in base_models.items():
        if name == "MLP":
            torch_full_model = train_torch_mlp(X_all_sc, y_all)
        elif name == "XGBoost":
            sw = compute_sample_weight("balanced", y_all)
            model.fit(X_all, y_all, sample_weight=sw)
        else:
            model.fit(X_all, y_all)
        print(f"    ✅ {name}")

    # Refit meta-learners on fresh OOF
    oof2 = np.zeros_like(oof_meta)
    for fi, (tr_i, val_i) in enumerate(TSCV.split(X_all)):
        Xtr2, Xval2 = X_all.iloc[tr_i], X_all.iloc[val_i]
        ytr2        = y_all.iloc[tr_i]
        sc2         = StandardScaler()
        Xtr2_sc  = pd.DataFrame(sc2.fit_transform(Xtr2),  columns=Xtr2.columns)
        Xval2_sc = pd.DataFrame(sc2.transform(Xval2),     columns=Xval2.columns)
        for mi, (name, model) in enumerate(base_models.items()):
            cs, ce = mi * 3, mi * 3 + 3
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

    oof_aug2    = np.hstack([oof2, X_all.values])
    mlp_meta_sc2 = StandardScaler()
    mlp_meta.fit(mlp_meta_sc2.fit_transform(oof_aug2), y_all)
    cat_meta.fit(oof_aug2, y_all)
    rf_meta.fit(oof_aug2, y_all)
    xgb_meta.fit(oof_aug2, y_all)

    # ── 4j. Holdout evaluation ────────────────────────────────────────────────
    X_test_hold   = feats_test[FEATURE_COLS]
    y_test_hold   = feats_test["Target"].values
    X_test_hold_sc = pd.DataFrame(
        sc_full.transform(X_test_hold), columns=X_test_hold.columns)
    elo_p_hold = get_elo_win_probs_vec(
        feats_test["EloH"].values, feats_test["EloA"].values)

    if len(X_test_hold) == 0:
        print("  ⚠️  No holdout data — skipping evaluation.")
        return None

    print(f"\n  📅 Evaluating {len(X_test_hold):,} holdout matches…\n")

    pred_rows            = []
    base_probs_hold: dict[str, np.ndarray] = {}

    # Base models (8)
    for name, model in base_models.items():
        p = (predict_torch_mlp(torch_full_model, X_test_hold_sc)
             if name == "MLP" else model.predict_proba(X_test_hold))
        base_probs_hold[name] = p
        pred_rows.append(_row(name, np.argmax(p, axis=1),
                              y_test_hold, probs=p))

    # Stacked ensembles (4)
    stk_m   = np.column_stack([base_probs_hold[n] for n in model_names])
    stk_aug = np.hstack([stk_m, X_test_hold.values])
    for mname, fn in [
        ("Stacked (CatMeta)",
         lambda: cat_meta.predict_proba(stk_aug)),
        ("Stacked (MLPMeta)",
         lambda: mlp_meta.predict_proba(
             mlp_meta_sc2.transform(stk_aug))),
        ("Stacked (RFMeta)",
         lambda: rf_meta.predict_proba(stk_aug)),
        ("Stacked (XGBMeta)",
         lambda: xgb_meta.predict_proba(stk_aug)),
    ]:
        p = fn()
        pred_rows.append(_row(mname, np.argmax(p, axis=1),
                              y_test_hold, probs=p))

    # Hybrid Elo-blended models (8)
    elo_summary_rows = []
    for hname, hw in hybrid_weights.items():
        bp = base_probs_hold.get(hname)
        if bp is None:
            continue
        hp  = hw["model_w"] * bp + hw["elo_w"] * elo_p_hold
        pred_rows.append(_row(f"Hybrid ({hname}-Elo)",
                              np.argmax(hp, axis=1),
                              y_test_hold, probs=hp))
        elo_summary_rows.append({
            "Scenario": scenario_name,
            "Base Model": hname,
            "model_w": hw["model_w"],
            "elo_w": hw["elo_w"],
        })

    # Draw-threshold variant (1)
    bp_best = base_probs_hold[best_ba]
    tp      = np.where(bp_best[:, 1] >= best_thresh, 1,
                       np.where(bp_best[:, 0] > bp_best[:, 2], 0, 2))
    pred_rows.append(_row(
        f"DrawThresh({best_thresh:.2f})+{best_ba}",
        tp, y_test_hold, probs=bp_best))

    # ── Build & print report ──────────────────────────────────────────────────
    report = (pd.DataFrame(pred_rows)
                .sort_values("pct_correct", ascending=False)
                .reset_index(drop=True))

    report.rename(columns={
        "pct_correct":   "% Correct",
        "correct":       "Correct",
        "home_recall":   "Home Recall %",
        "draw_recall":   "Draw Recall %",
        "away_recall":   "Away Recall %",
        "macro_f1":      "Macro-F1",
        "balanced_acc":  "Balanced Acc",
        "log_loss_soft": "Log Loss (soft)",
    }, inplace=True)

    print(f"  🏆 {scenario_name} — Holdout Results  "
          f"({len(report)} model variants)\n")
    print(report.to_string(index=True))
    best = report.iloc[0]
    print(f"\n  🥇 Best: {best['Model']}  |  "
          f"{best['Correct']}/{len(y_test_hold)} = {best['% Correct']}%  |  "
          f"Macro-F1: {best['Macro-F1']}  |  "
          f"Draw Recall: {best['Draw Recall %']}%  |  "
          f"Log Loss (soft): {best['Log Loss (soft)']}")

    # ── 4k. Save artefacts ────────────────────────────────────────────────────
    if save_models:
        save_dir = (f"saved_models_v5_"
                    f"{scenario_name.lower().replace(' ', '_')}")
        os.makedirs(save_dir, exist_ok=True)
        for name, model in base_models.items():
            if name == "MLP":
                torch.save(torch_full_model.state_dict(),
                           os.path.join(save_dir, "mlp_weights.pt"))
                pickle.dump(sc_full,
                            open(os.path.join(save_dir,
                                              "mlp_scaler.pkl"), "wb"))
            else:
                clean = name.lower().replace(" ", "_")
                pickle.dump(model,
                            open(os.path.join(save_dir,
                                              f"{clean}.pkl"), "wb"))
        json.dump(hybrid_weights,
                  open(os.path.join(save_dir,
                                    "hybrid_weights.json"), "w"), indent=2)
        report.to_csv(os.path.join(save_dir, "results_holdout.csv"),
                      index=False)
        pd.DataFrame(elo_summary_rows).to_csv(
            os.path.join(save_dir, "elo_weights_summary.csv"), index=False)
        print(f"  💾 Saved to '{save_dir}/'")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Ablation Study
# ─────────────────────────────────────────────────────────────────────────────
ABLATION_FEATURE_SETS = {
    "Odds only":        ODDS_COLS,
    "Elo only":         ELO_COLS,
    "Form only":        FORM_COLS,
    "Odds + Form":      ODDS_COLS + FORM_COLS,
    "Full ML features": FEATURE_COLS,
}


def run_ablation(scenario_name: str, div_filter: list | None):
    """
    Train a fixed LightGBM on each feature subset and evaluate on the
    2025/26 holdout.  Also runs the Full ML + Elo blend variant using
    the best hybrid weight found on the validation fold.
    """
    print(f"\n  🧪 Ablation study — {scenario_name}")

    if div_filter:
        mask      = df["Div"].isin(div_filter)
        cur_feats = feats_df[mask.values].copy()
    else:
        cur_feats = feats_df.copy()

    feats_train = cur_feats[cur_feats["Date"] < CUTOFF].copy()
    feats_test  = cur_feats[cur_feats["Date"] >= CUTOFF].copy()

    if len(feats_train) < 500 or len(feats_test) == 0:
        print("  ⚠️  Insufficient data — skipping ablation.")
        return None

    y_train = feats_train["Target"]
    y_test  = feats_test["Target"].values
    elo_p   = get_elo_win_probs_vec(
        feats_test["EloH"].values, feats_test["EloA"].values)

    tscv_abl = TimeSeriesSplit(n_splits=3)
    tr_last_abl, val_last_abl = list(tscv_abl.split(
        feats_train[FEATURE_COLS]))[-1]
    y_val_abl = feats_train["Target"].iloc[val_last_abl].values
    elo_p_val = get_elo_win_probs_vec(
        feats_train.iloc[val_last_abl]["EloH"].values,
        feats_train.iloc[val_last_abl]["EloA"].values)

    ablation_rows = []

    for label, cols in ABLATION_FEATURE_SETS.items():
        cols = [c for c in cols if c in feats_train.columns]
        if not cols:
            continue

        X_tr  = feats_train[cols]
        X_te  = feats_test[cols]

        lgb = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=63,
            class_weight="balanced", random_state=42, verbose=-1,
            device="gpu" if LGB_GPU else "cpu",
            n_jobs=1 if LGB_GPU else -1,
        )
        lgb.fit(X_tr, y_train)
        p    = lgb.predict_proba(X_te)
        preds = np.argmax(p, axis=1)
        s    = get_stats(preds, y_test, probs=p)
        ablation_rows.append({
            "Scenario":        scenario_name,
            "Feature Set":     label,
            "Accuracy %":      s["pct_correct"],
            "Macro-F1":        s["macro_f1"],
            "Balanced Acc":    s["balanced_acc"],
            "Home Recall %":   s["home_recall"],
            "Draw Recall %":   s["draw_recall"],
            "Away Recall %":   s["away_recall"],
            "Log Loss (soft)": s["log_loss_soft"],
        })
        print(f"    {label:<22}  "
              f"acc={s['pct_correct']:.2f}%  "
              f"macro-F1={s['macro_f1']:.4f}  "
              f"draw-R={s['draw_recall']:.1f}%")

    # Full ML + best Elo blend
    cols_full  = FEATURE_COLS
    X_tr_full  = feats_train[cols_full]
    X_te_full  = feats_test[cols_full]
    X_val_full = feats_train[cols_full].iloc[val_last_abl]

    lgb_full = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        class_weight="balanced", random_state=42, verbose=-1,
        device="gpu" if LGB_GPU else "cpu",
        n_jobs=1 if LGB_GPU else -1,
    )
    lgb_full.fit(X_tr_full, y_train)
    p_val  = lgb_full.predict_proba(X_val_full)
    wg     = np.round(np.arange(0.0, 1.01, 0.01), 2)
    blends = (wg[:, None, None] * p_val[None]
              + (1 - wg)[:, None, None] * elo_p_val[None])
    f1s    = np.array([f1_score(y_val_abl, np.argmax(b, axis=1),
                                average="macro") for b in blends])
    best_w = float(wg[np.argmax(f1s)])

    p_full  = lgb_full.predict_proba(X_te_full)
    hp      = best_w * p_full + (1 - best_w) * elo_p
    preds_h = np.argmax(hp, axis=1)
    s       = get_stats(preds_h, y_test, probs=hp)
    ablation_rows.append({
        "Scenario":        scenario_name,
        "Feature Set":     f"Full ML + Elo blend (w={best_w:.2f})",
        "Accuracy %":      s["pct_correct"],
        "Macro-F1":        s["macro_f1"],
        "Balanced Acc":    s["balanced_acc"],
        "Home Recall %":   s["home_recall"],
        "Draw Recall %":   s["draw_recall"],
        "Away Recall %":   s["away_recall"],
        "Log Loss (soft)": s["log_loss_soft"],
    })
    print(f"    {'Full ML + Elo blend':<22}  "
          f"acc={s['pct_correct']:.2f}%  "
          f"macro-F1={s['macro_f1']:.4f}  "
          f"draw-R={s['draw_recall']:.1f}%  "
          f"(elo_w={1-best_w:.2f})")

    abl_df = pd.DataFrame(ablation_rows)
    print(f"\n{abl_df.to_string(index=False)}")
    return abl_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Rolling-Origin Backtest  (full 21-variant pipeline per season)
# ─────────────────────────────────────────────────────────────────────────────
ROLLING_SEASONS = [
    ("2022/23", pd.Timestamp("2022-08-01"), pd.Timestamp("2023-08-01")),
    ("2023/24", pd.Timestamp("2023-08-01"), pd.Timestamp("2024-08-01")),
    ("2024/25", pd.Timestamp("2024-08-01"), pd.Timestamp("2025-08-01")),
]


def run_rolling_backtest(scenario_name: str, div_filter: list | None):
    """
    For each historical season in ROLLING_SEASONS, call run_pipeline() with
    train_cutoff=season_start and test_cutoff=season_end so that the full
    21-variant model set (8 base + 4 stacked + 8 hybrid-Elo + 1 draw-thresh)
    is trained and evaluated on every season window.

    Results are tagged with Scenario and Season columns and returned as a
    single concatenated DataFrame.  save_models=False to avoid cluttering disk
    with 15 extra model directories (5 leagues × 3 seasons).
    """
    print(f"\n{'='*65}")
    print(f"  📅  ROLLING-ORIGIN BACKTEST — {scenario_name}")
    print(f"      Full 21-variant pipeline per season")
    print(f"{'='*65}")

    season_frames = []

    for season_label, season_start, season_end in ROLLING_SEASONS:
        print(f"\n  ── Season {season_label}  "
              f"(train < {season_start.date()}  |  "
              f"test {season_start.date()} – {season_end.date()}) ──")

        rep = run_pipeline(
            scenario_name=f"{scenario_name} [{season_label}]",
            div_filter=div_filter,
            train_cutoff=season_start,
            test_cutoff=season_end,
            save_models=False,
        )

        if rep is None:
            print(f"  ⚠️  {season_label}: insufficient data — skipped.")
            continue

        rep.insert(0, "Season",   season_label)
        rep.insert(0, "Scenario", scenario_name)
        season_frames.append(rep)
        print(f"  ✅  {season_label} complete — "
              f"{len(rep)} model variants evaluated.")

    if not season_frames:
        print(f"  ⚠️  No valid seasons for {scenario_name}.")
        return None

    roll_df = pd.concat(season_frames, ignore_index=True)

    print(f"\n  📊 Rolling backtest summary — {scenario_name}  "
          f"({len(season_frames)} seasons × {len(rep)} variants = "
          f"{len(roll_df)} total rows)\n")
    print(roll_df[["Season", "Model", "% Correct", "Macro-F1",
                   "Balanced Acc", "Draw Recall %",
                   "Log Loss (soft)"]].to_string(index=False))
    return roll_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Scenario Definitions & Execution
# ─────────────────────────────────────────────────────────────────────────────
SCENARIOS = [
    {"name": "Premier League",         "filter": ["E0"]},
    {"name": "La Liga",                "filter": ["SP1"]},
    {"name": "Serie A",                "filter": ["I1"]},
    {"name": "Bundesliga",             "filter": ["D1"]},
    {"name": "Ligue 1",                "filter": ["F1"]},
    {"name": "Top 5 Leagues Combined", "filter": TOP5_DIV},
    {"name": "All Leagues Combined",   "filter": None},
]

# Top-5 individual leagues used for rolling backtest
ROLLING_SCENARIOS = [sc for sc in SCENARIOS
                     if sc["filter"] and len(sc["filter"]) == 1
                     and sc["filter"][0] in TOP5_DIV]

print(f"\n🚀 Running {len(SCENARIOS)} scenarios…  "
      f"Each produces ~21 model variants.\n")

all_results   = []
all_ablations = []
all_rolling   = []

for sc in SCENARIOS:
    # ── Main pipeline ─────────────────────────────────────────────────────────
    rep = run_pipeline(
        scenario_name=sc["name"],
        div_filter=sc["filter"],
        save_models=True,
    )
    if rep is not None:
        rep.insert(0, "Scenario", sc["name"])
        all_results.append(rep)

    # ── Ablation (all scenarios) ──────────────────────────────────────────────
    abl = run_ablation(
        scenario_name=sc["name"],
        div_filter=sc["filter"],
    )
    if abl is not None:
        all_ablations.append(abl)

# ── Rolling-origin backtest (individual top-5 leagues, full 21-variant) ───────
print("\n\n" + "="*65)
print("  📅  ROLLING-ORIGIN BACKTEST  (full 21-variant pipeline per season)")
print("="*65)
for sc in ROLLING_SCENARIOS:
    roll = run_rolling_backtest(
        scenario_name=sc["name"],
        div_filter=sc["filter"],
    )
    if roll is not None:
        all_rolling.append(roll)

# ── Aggregate saves ───────────────────────────────────────────────────────────
if all_results:
    pd.concat(all_results, ignore_index=True).to_csv(
        "all_scenarios_results.csv", index=False)
    print("\n💾 Saved: all_scenarios_results.csv")

if all_ablations:
    pd.concat(all_ablations, ignore_index=True).to_csv(
        "ablation_results.csv", index=False)
    print("💾 Saved: ablation_results.csv")

if all_rolling:
    pd.concat(all_rolling, ignore_index=True).to_csv(
        "rolling_backtest_results.csv", index=False)
    print("💾 Saved: rolling_backtest_results.csv")

print("\n🏁 All scenarios complete!")