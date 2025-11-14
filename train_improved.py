import pandas as pd
import numpy as np
import json
import joblib
import argparse
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# =========================================================
# 1) ELO SYSTEM
# =========================================================
def expected_result(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def update_elo(elo_home, elo_away, result, k=20, home_adv=60):
    # result: 1 = home win, 0 = draw, -1 = away win
    exp_home = expected_result(elo_home + home_adv, elo_away)
    exp_away = 1 - exp_home

    if result == 1:
        score_home, score_away = 1, 0
    elif result == 0:
        score_home, score_away = 0.5, 0.5
    else:
        score_home, score_away = 0, 1

    new_home = elo_home + k * (score_home - exp_home)
    new_away = elo_away + k * (score_away - exp_away)
    return new_home, new_away


# =========================================================
# 2) FORM & REST FEATURE GENERATION
# =========================================================
def last_n_points(results, n=5):
    # Results list: "H","D","A"
    if len(results) == 0:
        return 0
    pts = {"H": 3, "D": 1, "A": 0}
    last_results = results[-n:]
    return sum(pts[r] for r in last_results)


def last_n_goal_diff(goals_scored, goals_conceded, n=5):
    if len(goals_scored) == 0:
        return 0
    gs = goals_scored[-n:]
    gc = goals_conceded[-n:]
    return sum(gs) - sum(gc)


def compute_rest_days(current_date, previous_dates):
    if len(previous_dates) == 0:
        return 7
    diff = (current_date - previous_dates[-1]).days
    return max(1, diff)


# =========================================================
# MAIN TRAINING FUNCTION
# =========================================================
def train_model(csv_file):

    print("📂 Loading merged La Liga dataset...")
    df = pd.read_csv(csv_file)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    teams = sorted(list(set(df["HomeTeam"]).union(df["AwayTeam"])))

    # Initialize Elo dictionary
    elo = {team: 1500 for team in teams}

    # Tracking dictionaries
    team_history = {
        team: {
            "dates": [],
            "results": [],
            "goals_for": [],
            "goals_against": []
        }
        for team in teams
    }

    rows = []
    print("⚙️ Building training feature rows...")

    for idx, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        date = row["Date"]
        hg = row["FTHG"]
        ag = row["FTAG"]
        result_label = row["FTR"]

        # Convert result to numeric
        if result_label == "H":
            result = 1
        elif result_label == "D":
            result = 0
        else:
            result = -1

        # Pre-match Elo
        elo_home_before = elo[home]
        elo_away_before = elo[away]
        elo_diff = elo_home_before - elo_away_before + 60  # home advantage

        # Form stats
        home_hist = team_history[home]
        away_hist = team_history[away]

        home_pts5 = last_n_points(home_hist["results"])
        away_pts5 = last_n_points(away_hist["results"])

        home_gd5 = last_n_goal_diff(home_hist["goals_for"], home_hist["goals_against"])
        away_gd5 = last_n_goal_diff(away_hist["goals_for"], away_hist["goals_against"])

        home_rest = compute_rest_days(date, home_hist["dates"])
        away_rest = compute_rest_days(date, away_hist["dates"])

        # Store features
        rows.append({
            "Date": date,
            "HomeTeam": home,
            "AwayTeam": away,
            "EloHomeBefore": elo_home_before,
            "EloAwayBefore": elo_away_before,
            "EloDiff": elo_diff,
            "HomeFormPts5": home_pts5,
            "AwayFormPts5": away_pts5,
            "HomeGD5": home_gd5,
            "AwayGD5": away_gd5,
            "HomeRestDays": home_rest,
            "AwayRestDays": away_rest,
            "HomeWin": 1 if result == 1 else 0,
            "Draw": 1 if result == 0 else 0,
            "AwayWin": 1 if result == -1 else 0
        })

        # Update team histories
        home_hist["dates"].append(date)
        home_hist["results"].append("H" if result == 1 else ("D" if result == 0 else "A"))
        home_hist["goals_for"].append(hg)
        home_hist["goals_against"].append(ag)

        away_hist["dates"].append(date)
        away_hist["results"].append("A" if result == -1 else ("D" if result == 0 else "H"))
        away_hist["goals_for"].append(ag)
        away_hist["goals_against"].append(hg)

        # Update Elo
        new_home, new_away = update_elo(elo_home_before, elo_away_before, result)
        elo[home] = new_home
        elo[away] = new_away

    feats = pd.DataFrame(rows)
    feats.to_csv("models_improved/training_features.csv", index=False)

    # Save final Elo ratings
    elo_df = pd.DataFrame({"Team": list(elo.keys()), "Elo": list(elo.values())})
    elo_df.to_csv("models_improved/current_elo_ratings.csv", index=False)

    # Training labels + features
    feature_cols = [
        "EloHomeBefore", "EloAwayBefore", "EloDiff",
        "HomeFormPts5", "AwayFormPts5",
        "HomeGD5", "AwayGD5",
        "HomeRestDays", "AwayRestDays"
    ]

    X = feats[feature_cols]
    y = feats[["HomeWin", "Draw", "AwayWin"]].idxmax(axis=1).map(
        {"HomeWin": 0, "Draw": 1, "AwayWin": 2}
    )

    # =====================================================
    # Train XGBoost
    # =====================================================
    print("🔥 Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
    )
    xgb.fit(X, y)

    xgb_preds = xgb.predict(X)
    xgb_probs = xgb.predict_proba(X)
    xgb_acc = accuracy_score(y, xgb_preds)
    xgb_ll = log_loss(y, xgb_probs)

    joblib.dump(xgb, "models_improved/xgb_model.pkl")

    # =====================================================
    # Train Random Forest
    # =====================================================
    print("🌲 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=400, max_depth=20)
    rf.fit(X, y)

    rf_preds = rf.predict(X)
    rf_probs = rf.predict_proba(X)
    rf_acc = accuracy_score(y, rf_preds)
    rf_ll = log_loss(y, rf_probs)

    joblib.dump(rf, "models_improved/rf_model.pkl")

    # =====================================================
    # Save model parameters + metrics
    # =====================================================
    params = {
        "HOME_ADV": 60,
        "feature_cols": feature_cols,
        "metrics": {
            "XGBoost": {
                "accuracy": float(xgb_acc),
                "logloss": float(xgb_ll)
            },
            "Random Forest": {
                "accuracy": float(rf_acc),
                "logloss": float(rf_ll)
            }
        }
    }

    with open("models_improved/model_params.json", "w") as f:
        json.dump(params, f, indent=4)

    print("\n============================")
    print("TRAINING COMPLETE")
    print("============================")
    print(f"📈 XGBoost Accuracy: {xgb_acc:.4f}, LogLoss: {xgb_ll:.4f}")
    print(f"🌲 Random Forest Accuracy: {rf_acc:.4f}, LogLoss: {rf_ll:.4f}")
    print("============================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="Merged La Liga CSV file (laliga_merged_clean.csv)")
    args = parser.parse_args()
    train_model(args.csv_file)