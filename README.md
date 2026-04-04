# ⚽ La Liga Match Predictor Pro

An advanced, high-performance machine learning suite for predicting La Liga match outcomes. This project implements a **Hybrid AI Intelligence** framework, harmonizing long-term statistical team strength with high-velocity predictive analytics.

---

## 🧠 The "Hybrid Intelligence" Advantage

Traditional sports modeling often fails by being either too slow (Static Stats) or too reactive (Short-term ML). Our **Hybrid Stream** overcomes this by synthesizing two distinct analytical signatures:

### 1. 🛡️ Dynamic Elo Ratings (The Baseline)
- Tracks the **historical class** of every club across two decades of data.
- **Home Advantage Scaling**: Automatically injects a **+75 Elo point** offset for home teams.
- **Post-Match Decay**: Ratings evolved using an expected vs. actual score formula ($K=25$), ensuring the strength profile is always current.

### 2. ⚡ Multi-Model Gradient Boosting (The Momentum)
- **XGBoost, LightGBM, & CatBoost**: An ensemble of three distinct gradient-boosted architectures.
- **High-Dimensional Features**: Captures patterns in **Rolling Form (5-game window)**, **Goal Differential (GD)**, and **Market Consensus Odds**.
- **Market Intelligence**: Leverages bookmaker spreads as a high-signal feature for real-time sentiment.

> [!IMPORTANT]
> **🚀 Blended Intelligence Optimization**: The system performs an automated optimization loop to find the exact "Goldilocks" weights (e.g., _68% ML / 32% Elo_) that minimize Log Loss and maximize predictive accuracy for the current season's dynamics.

---

## 🛰️ Live Odds & Automation

This project isn't just about historical data. It's built for the **Live Season**:

- **Real-Time Data**: Integrates with **The Odds API** to fetch upcoming match spreads.
- **Automated Prediction**: One command (`python data2.py`) triggers a full cycle: Fetching -> Imputation -> Synthesis -> Backtesting -> Prediction.
- **Configuration**: Simply add your API key to `config.json` to unlock professional-grade market data.

---

## 🌟 Professional Features

- **🏆 Multi-Model Leaderboard**: Native support for XGBoost, CatBoost, LightGBM, Random Forest, and Logistic Regression.
- **⚖️ Dynamic Weighting**: Each season optimizes its own hybrid blend to match current league parity.
- **📈 Advanced Diagnostics**: High-resolution confusion matrices and interactive Plotly visualization.
- **🎨 Dark-Mode UI**: A premium Streamlit dashboard designed for clarity and visual impact.

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+**
- **pip**

### Installation

1.  **Clone the workspace**
    ```bash
    git clone https://github.com/yourusername/laliga-predictor.git
    cd laliga-predictor
    ```

2.  **Initialize Environment**
    ```bash
    pip install -r requirements.txt
    ```

### Execution Workflow

1.  **Run the Intelligence Pipeline**
    Calculates Elo, trains the ensemble, and optimizes hybrid weights.
    ```bash
    python data2.py
    ```

2.  **Launch the Dashboard**
    Interactive analytics at your fingertips.
    ```bash
    streamlit run hybrid_app.py
    ```

---

## 📁 Architecture Overview

```text
laliga-predictor/
├── hybrid_app.py                           # 🚀 Streamlit Entry (Analytics & Predictions)
├── data2.py                                # ⚙️ Intelligence Engine (Data -> Model)
├── data2.ipynb                             # 📝 Step-by-Step Methodology Documentation
├── laliga_merged_clean.csv                 # 📊 Cleaned Longitudinal Dataset
├── config.json                             # 🛰️ API Configuration (Odds API Key)
├── upcoming_odds.json                      # 📡 Live Fixture Odds (auto-updated)
├── current_team_stats.csv                  # 📈 Live Team Elo & Form Snapshot
├── saved_models_pkl/                       # 🧠 Trained Artifacts & Optimization Weights
│   ├── hybrid_xgboost_elo.pkl              # Top model pickle
│   ├── hybrid_xgboost_elo_weights.json     # ⚖️ Per-model blend weights (model_w, elo_w)
│   ├── model_metrics.json                  # 📊 Full Leaderboard Metadata
│   └── ...                                 # (1 .pkl + 1 _weights.json per top-3 model)
└── Match Data/                             # 📂 Local Cache of Raw Sources
```

---

## 🛠️ Technical Stack

- **Core**: Python 3.11+, Pandas, NumPy
- **ML Architecture**: XGBoost, LightGBM, CatBoost
- **Calibration**: Scikit-Learn (Probability Calibration)
- **UI & Viz**: Streamlit, Plotly (Interactive charts)
- **Data Hooks**: Football-Data.co.uk & The Odds API

---

## 📄 License

Licensed under the MIT License. Built for enthusiasts, data scientists, and analysts.

---

⚽ **Predict with Confidence. Analyze with Precision.** 🇪🇸
