# ⚽ La Liga Match Predictor (Elo + XGBoost Hybrid)

This project combines **Elo ratings** and **XGBoost machine learning** to predict outcomes of La Liga football matches.  
It uses historical match data to build dynamic team ratings and then trains an ML model to estimate the probability of **Home Win**, **Draw**, and **Away Win**.  
The app is deployed using **Streamlit** for interactive visualization.

---

## 🧩 How It Works

### 1️⃣ Data Preprocessing
The system takes a cleaned La Liga CSV (e.g., `laliga_merged_clean.csv`) containing:
- `Date`, `HomeTeam`, `AwayTeam`
- `FTHG` (Full-Time Home Goals), `FTAG` (Full-Time Away Goals)
- `FTR` (Full-Time Result: H/D/A)

It ensures consistent date formats and removes invalid or missing rows.

---

### 2️⃣ Elo Rating System
Each team begins with a **base Elo** of 1500.  
Elo ratings are updated after every match using:
- **K-factor:** 30  
- **Home advantage:** +90 Elo points

After each game:
- The **expected score** for each team is computed.  
- Elo ratings are adjusted based on the actual result:
  - Home win → Elo gain for home team
  - Away win → Elo gain for away team
  - Draw → minor adjustment for both

Elo data is stored per match as:
```
EloHomeBefore, EloAwayBefore, EloDiff
```

---

### 3️⃣ Recency Weighting
Recent matches influence the model more strongly than older ones.

A **time-decay weight** is applied:
$$
\text{Weight} = e^{-\frac{\text{days\_ago}}{180}}
$$
This gives exponentially higher weight to newer matches.

---

### 4️⃣ XGBoost Training
The **XGBoost classifier** predicts one of three outcomes:
- 0 → Home Win  
- 1 → Draw  
- 2 → Away Win  

Features used:
- `EloHomeBefore`
- `EloAwayBefore`
- `EloDiff`

Model configuration:
```python
xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    learning_rate=0.05,
    max_depth=5,
    n_estimators=400,
    subsample=0.8,
    colsample_bytree=0.8
)
```

Evaluation metrics:
- Accuracy
- Log Loss
- Classification Report

Trained artifacts are stored in `/models`:
```
xgb_model.pkl
training_features.csv
current_elo_ratings.csv
model_params.json
```

---

### 5️⃣ Streamlit Web App

The **Streamlit dashboard** (`app.py`) lets users:
1. Select Home and Away teams.
2. View **predicted probabilities** via bar chart or gauge chart.
3. See **Elo rating history trends** for both teams.
4. Retrain the model dynamically from the UI.

#### 📊 Components
- **Bar Chart** – shows win/draw probabilities.
- **Line Chart** – displays Elo rating progression.
- **Team Logos** (optional) – fetched from local folder or ESPN.
- **Retrain Button** – runs the pipeline again for updated data.

---

## 🗂️ Folder Structure

```
LaLiga-Match-Predictor/
│
├── train.py                 # Model training pipeline (Elo + XGBoost)
├── app.py                   # Streamlit UI for prediction
├── laliga_merged_clean.csv  # Historical dataset
├── requirements.txt         # Python dependencies
├── models/
│   ├── xgb_model.pkl
│   ├── training_features.csv
│   ├── current_elo_ratings.csv
│   └── model_params.json
└── README.md
```

---

## ⚙️ Installation & Run

### 1️⃣ Clone the Repo
```bash
git clone https://github.com/joel662/La-Liga-Match-Predictor-Elo-XGBoost-.git
cd La-Liga-Match-Predictor-Elo-XGBoost-
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model
```bash
python train.py laliga_merged_clean.csv
```

### 4️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 🌍 Optional Deployment
You can deploy this project for free using [Streamlit Cloud](https://share.streamlit.io):
1. Push your code to GitHub.
2. Go to **share.streamlit.io** → "New App".
3. Select this repo and choose `app.py`.
4. Click **Deploy** 🚀

---

## 🧠 Concept Summary
| Component | Description |
|------------|--------------|
| **Elo Rating** | Measures team strength dynamically. |
| **Recency Weight** | Recent games weigh more heavily. |
| **XGBoost Classifier** | Learns non-linear relationships between Elo and outcomes. |
| **Streamlit UI** | Interactive match predictor dashboard. |

---
