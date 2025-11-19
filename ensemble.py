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
    
    # Load model params to get feature columns and split info
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
    
    # Get train/test split info
    split_info = params.get("train_test_split", {})
    test_ratio = split_info.get("test_ratio", 0.2)
    split_idx = int(len(feats_df) * (1 - test_ratio))
    
    # Split data same way as training
    train_df = feats_df.iloc[:split_idx]
    test_df = feats_df.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df[["HomeWin", "Draw", "AwayWin"]].idxmax(axis=1).map(
        {"HomeWin": 0, "Draw": 1, "AwayWin": 2}
    )
    
    X_test = test_df[feature_cols]
    y_test = test_df[["HomeWin", "Draw", "AwayWin"]].idxmax(axis=1).map(
        {"HomeWin": 0, "Draw": 1, "AwayWin": 2}
    )
    
    # Create ensemble with optimized weights
    ensemble = EnsembleModel(
        xgb_model=xgb_model,
        rf_model=rf_model,
        weights=[0.55, 0.45]  # Slightly favor XGBoost
    )
    
    # Evaluate ensemble on training data
    print("📈 Evaluating ensemble model...")
    
    ensemble_train_preds = ensemble.predict(X_train)
    ensemble_train_probs = ensemble.predict_proba(X_train)
    ensemble_train_acc = accuracy_score(y_train, ensemble_train_preds)
    ensemble_train_ll = log_loss(y_train, ensemble_train_probs)
    
    ensemble_test_preds = ensemble.predict(X_test)
    ensemble_test_probs = ensemble.predict_proba(X_test)
    ensemble_test_acc = accuracy_score(y_test, ensemble_test_preds)
    ensemble_test_ll = log_loss(y_test, ensemble_test_probs)
    
    # Save ensemble model
    print("💾 Saving ensemble model...")
    joblib.dump(ensemble, f"{MODEL_DIR}/ensemble_model.pkl")
    
    # Update model_params.json with ensemble metrics
    print("📝 Updating model parameters with ensemble metrics...")
    params["metrics"]["Ensemble"] = {
        "train_accuracy": float(ensemble_train_acc),
        "train_logloss": float(ensemble_train_ll),
        "test_accuracy": float(ensemble_test_acc),
        "test_logloss": float(ensemble_test_ll),
        "accuracy": float(ensemble_test_acc),  # Use test accuracy for display
        "logloss": float(ensemble_test_ll)
    }
    
    # Add ensemble weights to params
    params["ensemble_weights"] = {
        "xgb": float(ensemble.weights[0]),
        "rf": float(ensemble.weights[1])
    }
    
    with open(f"{MODEL_DIR}/model_params.json", "w") as f:
        json.dump(params, f, indent=4)
    
    print("\n" + "="*60)
    print("ENSEMBLE BUILD COMPLETE")
    print("="*60)
    print(f"\n🎯 Ensemble Results:")
    print(f"   Train - Accuracy: {ensemble_train_acc:.4f}, LogLoss: {ensemble_train_ll:.4f}")
    print(f"   Test  - Accuracy: {ensemble_test_acc:.4f}, LogLoss: {ensemble_test_ll:.4f}")
    
    print(f"\n⚖️ Ensemble Weights:")
    print(f"   XGBoost: {ensemble.weights[0]:.2f}")
    print(f"   Random Forest: {ensemble.weights[1]:.2f}")
    
    # Compare with individual models
    if "XGBoost" in params["metrics"] and "Random Forest" in params["metrics"]:
        xgb_test_acc = params["metrics"]["XGBoost"]["test_accuracy"]
        rf_test_acc = params["metrics"]["Random Forest"]["test_accuracy"]
        
        print("\n" + "="*60)
        print("📊 Model Comparison (Test Set):")
        print("="*60)
        print(f"   XGBoost:       {xgb_test_acc:.4f}")
        print(f"   Random Forest: {rf_test_acc:.4f}")
        print(f"   Ensemble:      {ensemble_test_acc:.4f}")
        
        if ensemble_test_acc > max(xgb_test_acc, rf_test_acc):
            print("   🏆 Ensemble outperforms individual models!")
        elif ensemble_test_acc > min(xgb_test_acc, rf_test_acc):
            print("   ⚖️ Ensemble performs between the two models")
        else:
            print("   ⚠️ Consider adjusting ensemble weights")
        
        # Overfitting check
        ensemble_overfit = ensemble_train_acc - ensemble_test_acc
        print(f"\n📊 Overfitting Analysis:")
        print(f"   Ensemble gap: {ensemble_overfit:.4f} {'✅ Good' if ensemble_overfit < 0.1 else '⚠️ Overfitting'}")
        
    print("="*60)


if __name__ == "__main__":
    build_ensemble()