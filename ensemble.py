import pandas as pd
import numpy as np
import json
import joblib
from sklearn.metrics import accuracy_score, log_loss


class EnsembleModel:
    """
    Ensemble model that combines predictions from XGBoost and Random Forest
    using weighted averaging.
    """
    def __init__(self, xgb_model, rf_model, weights=None):
        self.xgb_model = xgb_model
        self.rf_model = rf_model
        # Default weights: slightly favor XGBoost
        self.weights = weights if weights is not None else [0.55, 0.45]
    
    def predict_proba(self, X):
        """
        Predict class probabilities using weighted average of both models.
        """
        xgb_probs = self.xgb_model.predict_proba(X)
        rf_probs = self.rf_model.predict_proba(X)
        
        # Weighted average
        ensemble_probs = (
            self.weights[0] * xgb_probs + 
            self.weights[1] * rf_probs
        )
        
        return ensemble_probs
    
    def predict(self, X):
        """
        Predict class labels.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


def build_ensemble():
    """
    Load trained models, create ensemble, evaluate, and save.
    """
    print("🔧 Building Ensemble Model...")
    
    MODEL_DIR = "models_improved"
    
    # Load trained models
    print("📦 Loading XGBoost and Random Forest models...")
    xgb_model = joblib.load(f"{MODEL_DIR}/xgb_model.pkl")
    rf_model = joblib.load(f"{MODEL_DIR}/rf_model.pkl")
    
    # Load training features for evaluation
    print("📊 Loading training features...")
    feats_df = pd.read_csv(f"{MODEL_DIR}/training_features.csv")
    
    # Load model params to get feature columns
    with open(f"{MODEL_DIR}/model_params.json") as f:
        params = json.load(f)
    
    feature_cols = params.get(
        "feature_cols",
        [
            "EloHomeBefore", "EloAwayBefore", "EloDiff",
            "HomeFormPts5", "AwayFormPts5",
            "HomeGD5", "AwayGD5",
            "HomeRestDays", "AwayRestDays"
        ]
    )
    
    # Prepare training data
    X = feats_df[feature_cols]
    y = feats_df[["HomeWin", "Draw", "AwayWin"]].idxmax(axis=1).map(
        {"HomeWin": 0, "Draw": 1, "AwayWin": 2}
    )
    
    # Create ensemble with optimized weights
    # You can experiment with different weights
    ensemble = EnsembleModel(
        xgb_model=xgb_model,
        rf_model=rf_model,
        weights=[0.55, 0.45]  # Slightly favor XGBoost
    )
    
    # Evaluate ensemble on training data
    print("📈 Evaluating ensemble model...")
    ensemble_preds = ensemble.predict(X)
    ensemble_probs = ensemble.predict_proba(X)
    
    ensemble_acc = accuracy_score(y, ensemble_preds)
    ensemble_ll = log_loss(y, ensemble_probs)
    
    print(f"✅ Ensemble Accuracy: {ensemble_acc:.4f}")
    print(f"✅ Ensemble Log Loss: {ensemble_ll:.4f}")
    
    # Save ensemble model
    print("💾 Saving ensemble model...")
    joblib.dump(ensemble, f"{MODEL_DIR}/ensemble_model.pkl")
    
    # Update model_params.json with ensemble metrics
    print("📝 Updating model parameters with ensemble metrics...")
    params["metrics"]["Ensemble"] = {
        "accuracy": float(ensemble_acc),
        "logloss": float(ensemble_ll)
    }
    
    # Add ensemble weights to params
    params["ensemble_weights"] = {
        "xgb": float(ensemble.weights[0]),
        "rf": float(ensemble.weights[1])
    }
    
    with open(f"{MODEL_DIR}/model_params.json", "w") as f:
        json.dump(params, f, indent=4)
    
    print("\n============================")
    print("ENSEMBLE BUILD COMPLETE")
    print("============================")
    print(f"📈 XGBoost Weight: {ensemble.weights[0]:.2f}")
    print(f"🌲 Random Forest Weight: {ensemble.weights[1]:.2f}")
    print(f"🎯 Ensemble Accuracy: {ensemble_acc:.4f}")
    print(f"📉 Ensemble Log Loss: {ensemble_ll:.4f}")
    print("============================")
    
    # Compare with individual models
    if "XGBoost" in params["metrics"] and "Random Forest" in params["metrics"]:
        xgb_acc = params["metrics"]["XGBoost"]["accuracy"]
        rf_acc = params["metrics"]["Random Forest"]["accuracy"]
        
        print("\n📊 Model Comparison:")
        print(f"   XGBoost:       {xgb_acc:.4f}")
        print(f"   Random Forest: {rf_acc:.4f}")
        print(f"   Ensemble:      {ensemble_acc:.4f}")
        
        if ensemble_acc > max(xgb_acc, rf_acc):
            print("   🏆 Ensemble outperforms individual models!")
        elif ensemble_acc > min(xgb_acc, rf_acc):
            print("   ⚖️ Ensemble performs between the two models")
        else:
            print("   ⚠️ Consider adjusting ensemble weights")


if __name__ == "__main__":
    build_ensemble()