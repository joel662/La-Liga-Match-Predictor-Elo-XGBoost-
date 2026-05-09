# A Hybrid Elo-Machine Learning Approach to Football Match Prediction

**Joel Varghese, Rufat Abdulzada, Md Zamilur Rahman**  
Algoma University · University of Windsor · NORDIK Institute

> Code repository for the paper submitted to CCIDSA 2026 (Springer STEAM-H proceedings).  
> Paper URL: *(to be added upon acceptance)*

---

## Overview

This repository contains all code to reproduce the experiments in the paper. We benchmark hybrid Elo-machine learning architectures for three-outcome football match prediction (home win, draw, away win) across 114,305 historical matches from 31 leagues (2003–2026). Eight base classifiers are evaluated across seven modelling scenarios, with stacked meta-learning, draw-threshold optimisation, Elo blending, feature ablation, and a rolling-origin backtest.

An interactive Streamlit web app (`app.py`) is also included, allowing users to upload their own data and predict any fixture using any trained model.

---

## Repository Structure

```
Football-Match-Prediction/
│
├── data3_advanced_v5.py        # Main experiment script — all paper results
├── app.py                      # Streamlit web app for interactive prediction
├── add_leagues.py              # Utility: download and merge league CSV files
├── Test.py                     # Quick sanity-check / smoke test
│
├── Match Data/                 # Per-league raw CSV files (from football-data.co.uk)
├── ALLleagues/                 # Intermediate per-league processed files
│
├── all_leagues.csv             # Merged dataset — input to data3_advanced_v5.py
├── all_scenarios_results.csv   # Full holdout results across all 7 scenarios (Tables 3–9)
├── ablation_results.csv        # Feature ablation raw output (Table 11)
├── ablation_table.csv          # Formatted ablation table (Table 11)
├── rolling_backtest_results.csv # Rolling-origin backtest output (Table 12)
├── current_team_stats.csv      # Latest Elo ratings and team statistics
│
├── ablation_chart.pdf/.png     # Ablation visualisation figure
├── acc_vs_draw_recall.pdf/.png # Accuracy vs draw recall trade-off figure
├── best_accuracy.pdf/.png      # Best accuracy per scenario figure
├── rolling_stability.pdf/.png  # Rolling backtest stability figure
│
├── config.json                 # Scenario and hyperparameter configuration
├── catboost_info/              # CatBoost training artefacts (auto-generated)
├── .devcontainer/              # Dev container configuration
├── .gitignore
└── README.md
```

---

## Requirements

Python 3.10+ is recommended. Install all dependencies with:

```bash
pip install -r requirements.txt
```

**Key dependencies:**

| Package | Purpose |
|---|---|
| scikit-learn | RF, SVM, LR, HistGBM, calibration, stacking, metrics |
| xgboost | XGBoost classifier |
| lightgbm | LightGBM classifier |
| catboost | CatBoost classifier |
| torch | PyTorch MLP (Sections 3.5, 4) |
| optuna | Hyperparameter tuning (Section 3.6) |
| pandas, numpy | Data loading and manipulation |
| streamlit | Interactive web app (`app.py`) |
| tqdm | Progress bars |

A CUDA-capable GPU is optional but strongly recommended — tested on an **NVIDIA RTX 4060 Laptop GPU, CUDA 12.1**. The script auto-detects GPU availability and falls back to CPU gracefully. Expect significantly longer runtimes on CPU, particularly for the rolling backtest.

---

## Data

Match data is sourced from [football-data.co.uk](https://www.football-data.co.uk/). The merged dataset `all_leagues.csv` covers 31 leagues from 2003/04 to 2025/26.

To rebuild `all_leagues.csv` from scratch:

```bash
python add_leagues.py
```

This downloads per-league CSVs and merges them into a single file. Alternatively, you can use the pre-built `all_leagues.csv` already in the repository.

**Required columns:** `Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`, `Div`, plus optional odds columns (`B365H/D/A`, `PSH/D/A`, `BbAvH/D/A`).

> **Important:** The 2025/26 season is used exclusively as the held-out test set and is never touched during training or validation.

---

## Reproducing Paper Results

All seven scenarios, the ablation study, and the rolling backtest are run by a single command:

```bash
python data3_advanced_v5.py
```

The script is structured into clearly labelled sections:

| Section | Description |
|---|---|
| `SECTION 0` | Imports and GPU detection |
| `SECTION 1` | Data loading and odds imputation |
| `SECTION 2` | Feature engineering over the full dataset (39 features) |
| `SECTION 3` | PyTorch MLP definition and training utilities |
| `SECTION 4` | Core `run_pipeline()` — Optuna tuning, OOF stacking, Elo blending, draw-threshold, holdout evaluation |
| `SECTION 5` | `run_ablation()` — feature group ablation study |
| `SECTION 6` | Rolling-origin backtest (3 historical seasons × 5 leagues) |
| `SECTION 7` | Scenario definitions and execution |

> ⚠️ **Runtime warning:** The full pipeline runs 50 Optuna trials per model per scenario. The rolling backtest runs the complete 21-model pipeline independently for 3 seasons × 5 leagues. On GPU this takes several hours; on CPU expect overnight runtimes.

---

## Table & Figure Index

Every table and figure in the paper maps to a specific script and output file:

| Paper Item | Description | Produced by | Output file |
|---|---|---|---|
| Table 1 | Dataset splits per scenario | `data3_advanced_v5.py` §1 | *(printed to console)* |
| Table 2 | Optuna hyperparameter search spaces | `data3_advanced_v5.py` §4b | *(documented in code)* |
| Table 3 | Premier League holdout results | `run_pipeline("Premier League", ["E0"])` | `saved_models_v5_premier_league/results_holdout.csv` |
| Table 4 | La Liga holdout results | `run_pipeline("La Liga", ["SP1"])` | `saved_models_v5_la_liga/results_holdout.csv` |
| Table 5 | Serie A holdout results | `run_pipeline("Serie A", ["I1"])` | `saved_models_v5_serie_a/results_holdout.csv` |
| Table 6 | Bundesliga holdout results | `run_pipeline("Bundesliga", ["D1"])` | `saved_models_v5_bundesliga/results_holdout.csv` |
| Table 7 | Ligue 1 holdout results | `run_pipeline("Ligue 1", ["F1"])` | `saved_models_v5_ligue_1/results_holdout.csv` |
| Table 8 | Top-5 Combined holdout results | `run_pipeline("Top 5 Combined", TOP5_DIV)` | `saved_models_v5_top_5_combined/results_holdout.csv` |
| Table 9 | All Leagues Combined holdout results | `run_pipeline("All Leagues", None)` | `saved_models_v5_all_leagues/results_holdout.csv` |
| Table 10 | Cross-league summary (best model per scenario) | Aggregated from all `run_pipeline()` calls | `all_scenarios_results.csv` |
| Table 11 | Feature ablation study | `run_ablation()` §5 | `ablation_results.csv`, `ablation_table.csv` |
| Table 12 | Rolling-origin backtest | Rolling backtest loop §6 | `rolling_backtest_results.csv` |
| Ablation figure | Accuracy by feature subset | `data3_advanced_v5.py` | `ablation_chart.pdf` / `ablation_chart.png` |
| Acc vs Draw Recall | Precision-recall trade-off | `data3_advanced_v5.py` | `acc_vs_draw_recall.pdf` / `acc_vs_draw_recall.png` |
| Best accuracy figure | Best model accuracy per scenario | `data3_advanced_v5.py` | `best_accuracy.pdf` / `best_accuracy.png` |
| Rolling stability figure | Accuracy stability over seasons | `data3_advanced_v5.py` | `rolling_stability.pdf` / `rolling_stability.png` |

Elo blend weights per model per scenario are saved to:
```
saved_models_v5_<scenario>/elo_weights_summary.csv
```

---

## Interactive Web App

A Streamlit app is included for exploring predictions without running the full pipeline:

```bash
streamlit run app.py
```

Upload `all_leagues.csv`, select a league scenario and model, optionally enter bookmaker odds, and predict any fixture. Models are trained on first run and cached — subsequent predictions are fast. The app displays win probabilities, Elo baseline comparison, and a value bet edge analysis.

---

## Saved Model Artefacts

After running `data3_advanced_v5.py`, trained models are saved per scenario to:

```
saved_models_v5_<scenario_name>/
├── logistic_regression.pkl
├── random_forest.pkl
├── linear_svm.pkl
├── xgboost.pkl
├── lightgbm.pkl
├── catboost.pkl
├── histgbm.pkl
├── mlp_weights.pt           # PyTorch MLP state dict
├── mlp_scaler.pkl           # StandardScaler used for MLP inputs
├── hybrid_weights.json      # Optimal Elo blend weight per base model
├── elo_weights_summary.csv  # Elo weight summary table
└── results_holdout.csv      # Full holdout results for this scenario
```

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@inproceedings{varghese2026hybrid,
  title     = {A Hybrid Elo-Machine Learning Approach to Football Match Prediction},
  author    = {Varghese, Joel and Abdulzada, Rufat and Rahman, Md Zamilur},
  booktitle = {Proceedings of the 2nd International Conference on Cognitive Computing,
               Intelligence and Data Science Applications (CCIDSA 2026)},
  series    = {STEAM-H: Science, Technology, Engineering, Agriculture, Mathematics \& Health},
  publisher = {Springer},
  year      = {2026}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
