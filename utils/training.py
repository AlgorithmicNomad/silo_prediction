import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

def train_model(model_name, X_train, y_train):
    """Trains a specified ML model."""
    if model_name == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == 'Gradient Boosting':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_name == 'XGBoost':
        if XGBRegressor is None:
            raise ImportError(
                "XGBoost is not installed in the active Python environment. "
                "Install project requirements or choose a different model."
            )
        model = XGBRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and returns metrics and predictions."""
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }
    return metrics, predictions

def save_model(model, scaler, model_name):
    """Saves the trained model and scaler to disk."""
    os.makedirs('models/trained_models', exist_ok=True)
    clean_name = model_name.replace(' ', '_').lower()
    
    model_path = f'models/trained_models/{clean_name}_model.pkl'
    scaler_path = f'models/trained_models/scaler.pkl'
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    return model_path

def load_model(model_name):
    """Loads a trained model and scaler from disk."""
    clean_name = model_name.replace(' ', '_').lower()
    model_path = f'models/trained_models/{clean_name}_model.pkl'
    scaler_path = f'models/trained_models/scaler.pkl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None
