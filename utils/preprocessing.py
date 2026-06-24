import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_data(filepath='dataset/generated_dataset.csv'):
    """Loads the generated dataset from CSV."""
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None

def prepare_data_for_training(df, target_col='Hoop_Stress_sigma_h'):
    """Prepares and scales data for machine learning models."""
    # We drop the target column. We can also choose to drop intermediate derived 
    # columns like 'Lateral_Pressure_p' to force the model to learn from base params.
    if 'Lateral_Pressure_p' in df.columns:
        X = df.drop(columns=[target_col, 'Lateral_Pressure_p'])
    else:
        X = df.drop(columns=[target_col])
        
    y = df[target_col]
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns
