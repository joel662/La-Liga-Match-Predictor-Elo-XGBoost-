# ⚽ La Liga Match Predictor

An advanced machine learning system for predicting La Liga match outcomes using hybrid Elo ratings and ensemble modeling.



## 🌟 Features

- **Hybrid Elo System** - Dynamic team ratings that update after each match
- **Multiple ML Models** - XGBoost, Random Forest, and Ensemble predictions
- **Real-time Updates** - Fetch latest La Liga data and retrain models on-demand
- **Interactive Dashboard** - Beautiful Streamlit UI with live predictions
- **Historical Analysis** - Track Elo rating trends over time
- **Performance Metrics** - View accuracy and log loss for each model

## 📊 Model Performance

The system uses three prediction models:

- **XGBoost** - Gradient boosting with optimized hyperparameters
- **Random Forest** - Ensemble of decision trees
- **Ensemble** - Weighted combination of both models (typically best performance)

Each model is trained on features including:
- Pre-match Elo ratings
- Recent form (last 5 matches)
- Goal differential trends
- Rest days between matches
- Home advantage factor

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/laliga-predictor.git
cd laliga-predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up directory structure**
```bash
mkdir -p "Match Data"
mkdir -p models_improved
```

### Initial Setup

1. **Download historical data**
   - Place La Liga CSV files in the `Match Data/` directory
   - Files should be from [football-data.co.uk](https://www.football-data.co.uk/)

2. **Merge datasets**
```bash
python merge.py
```

3. **Train initial models**
```bash
python train_improved.py laliga_merged_clean.csv
```

4. **Build ensemble**
```bash
python ensemble.py
```

5. **Launch the app**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
laliga-predictor/
├── app.py                          # Streamlit web application
├── train_improved.py               # Model training script
├── ensemble.py                     # Ensemble model builder
├── merge.py                        # Data merging utility
├── requirements.txt                # Python dependencies
├── Match Data/                     # Raw CSV files
│   └── SP1_*.csv
├── models_improved/                # Trained models & artifacts
│   ├── xgb_model.pkl              # XGBoost model
│   ├── rf_model.pkl               # Random Forest model
│   ├── ensemble_model.pkl         # Ensemble model
│   ├── current_elo_ratings.csv    # Latest Elo ratings
│   ├── training_features.csv      # Feature dataset
│   └── model_params.json          # Model metadata & metrics
└── laliga_merged_clean.csv        # Merged training dataset
```

## 🎯 Usage

### Making Predictions

1. **Select a model** from the dropdown (XGBoost, Random Forest, or Ensemble)
2. **Choose teams** - Select home and away teams
3. **View predictions** - See win/draw/loss probabilities
4. **Analyze trends** - Check historical Elo ratings

### Updating Models

Click the **"Fetch Latest Data & Retrain"** button to:
- Download the latest La Liga data
- Merge with historical data
- Retrain all models
- Update the ensemble

## 🔧 Configuration

### Model Parameters

Edit `train_improved.py` to adjust:

```python
# XGBoost parameters
xgb = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
)

# Random Forest parameters
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=20
)
```

### Elo System

Adjust Elo parameters in `train_improved.py`:

```python
def update_elo(elo_home, elo_away, result, k=20, home_adv=60):
    # k: learning rate (higher = more volatile)
    # home_adv: home advantage points
```

### Ensemble Weights

Modify ensemble weights in `ensemble.py`:

```python
ensemble = EnsembleModel(
    xgb_model=xgb_model,
    rf_model=rf_model,
    weights=[0.55, 0.45]  # [XGBoost weight, RF weight]
)
```

## 📈 Model Training Details

### Feature Engineering

The system generates the following features for each match:

| Feature | Description |
|---------|-------------|
| `EloHomeBefore` | Home team's Elo rating before match |
| `EloAwayBefore` | Away team's Elo rating before match |
| `EloDiff` | Elo difference + home advantage |
| `HomeFormPts5` | Home team's points in last 5 matches |
| `AwayFormPts5` | Away team's points in last 5 matches |
| `HomeGD5` | Home team's goal difference (last 5) |
| `AwayGD5` | Away team's goal difference (last 5) |
| `HomeRestDays` | Days since home team's last match |
| `AwayRestDays` | Days since away team's last match |

### Elo Rating System

- **Initial Rating**: 1500 for all teams
- **K-factor**: 20 (controls rating volatility)
- **Home Advantage**: +60 Elo points
- **Update Formula**: Based on expected vs actual results

## 📊 Performance Metrics

Models are evaluated using:

- **Accuracy** - Percentage of correct predictions
- **Log Loss** - Probabilistic prediction quality (lower is better)

Typical performance:
- Accuracy: 50-55%
- Log Loss: 1.00-1.10

## 🛠️ Technical Stack

- **Python 3.8+** - Core language
- **Streamlit** - Web interface
- **XGBoost** - Gradient boosting
- **Scikit-learn** - Random Forest & metrics
- **Pandas** - Data manipulation
- **Plotly** - Interactive charts
- **Joblib** - Model serialization

## 📝 Requirements

```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
plotly>=5.14.0
joblib>=1.2.0
requests>=2.28.0
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Data source: [Football-Data.co.uk](https://www.football-data.co.uk/)
- Elo rating system inspired by chess rankings
- Built with ❤️ for football analytics enthusiasts

## 📧 Contact

**Developer**: Joel

For questions or suggestions, please open an issue on GitHub.

---

⚽ **Enjoy predicting La Liga matches!** 🇪🇸