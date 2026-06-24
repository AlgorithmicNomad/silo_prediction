import os
import pandas as pd
import sys
from pathlib import Path
import joblib

# Robust path handling
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from utils.physics import generate_synthetic_data
from utils.preprocessing import prepare_data_for_training
from utils.training import train_model, evaluate_model, save_model

def main():
    print("🚀 Initializing Silo-ML Artifact Generation...")
    
    # Ensure directories exist
    (BASE_DIR / "dataset").mkdir(exist_ok=True)
    (BASE_DIR / "models" / "trained_models").mkdir(parents=True, exist_ok=True)
    
    # 1. Generate Dataset
    print("📦 Generating synthetic dataset (10,000 samples)...")
    df = generate_synthetic_data(num_samples=10000, add_noise=True)
    dataset_path = BASE_DIR / "dataset" / "generated_dataset.csv"
    df.to_csv(dataset_path, index=False)
    
    # 2. Train Base Models
    print("🧠 Training baseline ML models...")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data_for_training(df, "Hoop_Stress_sigma_h")
    
    for model_name in ["XGBoost", "Random Forest"]:
        print(f"   - Training {model_name}...")
        model = train_model(model_name, X_train, y_train)
        metrics, predictions = evaluate_model(model, X_test, y_test)
        save_model(model, scaler, model_name)
        
        # Save artifacts
        metrics_path = BASE_DIR / "models" / "trained_models" / f'metrics_{model_name.replace(" ", "_")}.pkl'
        joblib.dump(metrics, metrics_path)
        
        pred_path = BASE_DIR / "dataset" / f'predictions_{model_name.replace(" ", "_")}.csv'
        pd.Series(predictions, name='Predicted').to_csv(pred_path, index=False)
        print(f"     ✅ Done. R2: {metrics['R2']:.4f}")

    # Save shared artifacts
    pd.DataFrame(X_test, columns=feature_names).to_csv(BASE_DIR / "dataset" / "X_test.csv", index=False)
    pd.Series(y_test, name='Actual').to_csv(BASE_DIR / "dataset" / "y_test.csv", index=False)
    pd.Series(feature_names).to_csv(BASE_DIR / "dataset" / "feature_names.csv", index=False)

    print("\n✨ All artifacts generated successfully!")

if __name__ == "__main__":
    main()
