import pandas as pd
import numpy as np
from datetime import datetime
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, log_loss
import joblib
import argparse
import os

# ---------- Elo Parameters ----------
BASE_ELO = 1500
K = 30
HOME_ADV = 90

def expected_score(Ra, Rb):
    return 1 / (1 + 10 ** ((Rb - Ra) / 400))

def update_elo(Ra, Rb, Sa, Sb, home_adv=0):
    Ea = expected_score(Ra + home_adv, Rb)
    Eb = expected_score(Rb, Ra + home_adv)
    Ra_new = Ra + K * (Sa - Ea)
    Rb_new = Rb + K * (Sb - Eb)
    return Ra_new, Rb_new

def result_value(hg, ag):
    if hg > ag: return 1, 0
    if hg == ag: return 0.5, 0.5
    return 0, 1

def preprocess(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df = df.dropna(subset=["HomeTeam","AwayTeam","FTHG","FTAG"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    return df

def build_elo_features(df):
    teams = pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]])).tolist()
    ratings = {t: BASE_ELO for t in teams}
    feats = []

    for _, row in df.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        hg, ag = row["FTHG"], row["FTAG"]
        Rh, Ra = ratings[h], ratings[a]
        Sh, Sa = result_value(hg, ag)
        Rh_new, Ra_new = update_elo(Rh, Ra, Sh, Sa, HOME_ADV)

        feats.append({
            "Date": row["Date"], "HomeTeam": h, "AwayTeam": a,
            "FTHG": hg, "FTAG": ag, "Result": row["FTR"],
            "EloHomeBefore": Rh, "EloAwayBefore": Ra,
            "EloDiff": Rh - Ra + HOME_ADV,
            "HomeWin": 1 if hg>ag else 0,
            "Draw": 1 if hg==ag else 0,
            "AwayWin": 1 if hg<ag else 0
        })

        ratings[h], ratings[a] = Rh_new, Ra_new

    feats_df = pd.DataFrame(feats)
    ratings_df = pd.DataFrame([
        {"Team": t, "Elo": r} for t, r in ratings.items()
    ]).sort_values("Elo", ascending=False)
    return feats_df, ratings_df

def add_recency_weight(df):
    latest = df["Date"].max()
    days = (latest - df["Date"]).dt.days
    df["Weight"] = np.exp(-days / 180)
    return df

def train_xgboost(df):
    X = df[["EloHomeBefore","EloAwayBefore","EloDiff"]]
    y = df[["HomeWin","Draw","AwayWin"]].idxmax(axis=1).map({
        "HomeWin":0,"Draw":1,"AwayWin":2
    })
    X_train,X_test,y_train,y_test,w_train,w_test = train_test_split(
        X, y, df["Weight"], test_size=0.2, random_state=42, stratify=y
    )
    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", learning_rate=0.05,
        max_depth=5, n_estimators=400, subsample=0.8, colsample_bytree=0.8
    )
    model.fit(X_train, y_train, sample_weight=w_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    print("\n✅ Accuracy:", round(accuracy_score(y_test,preds),3))
    print("LogLoss:", round(log_loss(y_test,probs),3))
    print(classification_report(y_test,preds))
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_file", help="Cleaned LaLiga CSV")
    args = ap.parse_args()

    print("📂 Loading:", args.csv_file)
    df = pd.read_csv(args.csv_file)
    df = preprocess(df)
    print("⚙️ Computing Elo features...")
    feats_df, ratings_df = build_elo_features(df)
    feats_df = add_recency_weight(feats_df)

    print("🚀 Training XGBoost...")
    model = train_xgboost(feats_df)

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgb_model.pkl")
    feats_df.to_csv("models/training_features.csv", index=False)
    ratings_df.to_csv("models/current_elo_ratings.csv", index=False)
    with open("models/model_params.json","w") as f:
        json.dump({"BASE_ELO":BASE_ELO,"K":K,"HOME_ADV":HOME_ADV},f)

    print("\n✅ Training Complete. Artifacts saved in /models.")

if __name__=="__main__":
    main()
