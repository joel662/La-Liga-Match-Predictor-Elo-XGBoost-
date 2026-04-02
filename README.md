# ⚽ La Liga Match Predictor Pro

An advanced, next-gen machine learning system for predicting La Liga match outcomes. This project leverages a **Hybrid Elo-ML** approach to synthesize historical team strength with modern predictive analytics.

---

## 🧩 What is "Hybrid" Modeling?

In this system, **Hybrid** refers to the dual-stream predictive engine that merges two distinct methodologies:

1.  **Dynamic Elo Ratings (The Baseline)**:
    - Represents the historical strength of teams based on their entire match history.
    - Includes a **Standard Home Advantage (+75 Elo points)**.
    - Updated post-match using an expected vs. actual score formula ($K=25$).
    - Provides a stable, long-term probabilistic baseline for every matchup.

2.  **Machine Learning Classifiers (The Momentum)**:
    - Advanced models (**XGBoost**, **LightGBM**, **CatBoost**, **Random Forest**) analyze high-dimensional features.
    - Captures non-linear patterns like **Recent Form (last 5 games)**, **Goal Differentials (GD)**, and **Market Odds (Bookmaker Consensus)**.
    - Reacts quickly to tactical shifts and squad changes that Elo might miss.

> [!TIP]
> **The Synthesis**: The final prediction is a weighted average of both streams. The system automatically optimizes these "Blended Intelligence" weights (e.g., _70% ML / 30% Elo_) by testing thousands of combinations against a validation set to minimize Log Loss.

---

## 🌟 Features

- **🏆 Multi-Model Leaderboard** - Comparison of XGBoost, CatBoost, LightGBM, and Random Forest.
- **⚖️ Optimal Blending** - Automated weight optimization for every model configuration.
- **📈 Deep Analytics** - Interactive confusion matrices and feature importance pie charts.
- **🔄 Unified Pipeline** - Single script (`data2.py`) for data fetching, cleaning, Elo calculation, and model training.
- **🎨 Premium UI** - Stunning Streamlit dashboard with dark-mode aesthetics and real-time visualization.

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip
```

### Installation

1.  **Clone the repository**

    ```bash
    git clone https://github.com/yourusername/laliga-predictor.git
    cd laliga-predictor
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### Running the System

The pipeline is fully automated. You only need two commands:

1.  **Run the Training Pipeline**
    This script downloads latest data, calculates Elo ratings, trains all models, and saves the top 3 performers.

    ```bash
    python data2.py
    ```

2.  **Launch the Dashboard**
    ```bash
    streamlit run hybrid_app.py
    ```

---

## 📁 Project Structure

```text
laliga-predictor/
├── hybrid_app.py           # 🚀 Main Entry Point (Streamlit Dashboard)
├── data2.py                # ⚙️ Unified Pipeline (Data -> Elo -> Training)
├── laliga_merged_clean.csv # 📊 Integrated Master Dataset
├── current_team_stats.csv  # 📈 Live stats for dashboard injection
├── saved_models_pkl/       # 🧠 Trained artifacts & optimization weights
│   ├── xgboost.pkl
│   ├── xgboost_weights.json
│   └── model_metrics.json  # 📊 Leaderboard & accuracy metadata
├── Match Data/             # 📂 Local cache of raw CSV sources
└── requirements.txt        # 📝 Project dependencies
```

---

## 📊 Model Training Details

### Feature Set

The system trains on a curated subset of high-signal features:

- `EloDiff`: The strength gap between teams (including Home Advantage).
- `H_GD` / `A_GD`: Rolling goal differential for Home and Away teams.
- `Market_H` / `Market_D` / `Market_A`: Aggregated bookmaker consensus (Market Intelligence).

### Training Strategy

- **Chronological Split**: 80/20 train-test split based on time to prevent data leakage.
- **Probability Synthesis**: Soft-thresholding of ML probabilities blended with Elo expectations.
- **Log Loss Optimization**: Models are tuned to maximize confidence while minimizing error.

---

## 🛠️ Technical Stack

- **Logic**: Python 3.11+, Pandas, NumPy
- **Predictive Engine**: XGBoost, LightGBM, CatBoost, Scikit-learn
- **Frontend**: Streamlit, Plotly (Interactive Charts)
- **Data Source**: Football-Data.co.uk

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1.  Fork the repository
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⚽ **Enjoy predicting La Liga matches!** 🇪🇸
