import streamlit as st
import sys
import os
import joblib
import pandas as pd
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

try:
    from utils.preprocessing import load_data, prepare_data_for_training
    from utils.training import train_model, evaluate_model, save_model
except ImportError as e:
    st.error(f"Required utility modules could not be loaded: {e}")
    st.stop()

st.set_page_config(page_title="ML Training", page_icon="🧠")

st.title("🧠 Machine Learning Model Training")

df = load_data()

if df is None:
    st.warning("Dataset not found. Please go to 'Dataset Generator' and generate a dataset first.")
else:
    st.write(f"Dataset loaded with {len(df)} samples.")
    
    with st.sidebar:
        st.header("Training Configuration")
        selected_model = st.selectbox(
            "Select Model Architecture",
            ["Random Forest", "Gradient Boosting", "XGBoost"]
        )
        target_variable = "Hoop_Stress_sigma_h"
    
    if st.button(f"Train {selected_model} Model", type="primary"):
        with st.spinner("Preparing data..."):
            X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data_for_training(df, target_variable)
            
        with st.spinner(f"Training {selected_model}..."):
            model = train_model(selected_model, X_train, y_train)
            
        with st.spinner("Evaluating model..."):
            metrics, predictions = evaluate_model(model, X_test, y_test)
            
            # Save the model
            save_model(model, scaler, selected_model)
            
            # Save artifacts for visualization/comparison
            os.makedirs('dataset', exist_ok=True)
            pd.DataFrame(X_test, columns=feature_names).to_csv('dataset/X_test.csv', index=False)
            pd.Series(y_test, name='Actual').to_csv('dataset/y_test.csv', index=False)
            pd.Series(predictions, name='Predicted').to_csv(f'dataset/predictions_{selected_model.replace(" ", "_")}.csv', index=False)
            pd.DataFrame({
                'Actual': y_test,
                'Predicted': predictions,
            }).to_csv(f'dataset/evaluation_{selected_model.replace(" ", "_")}.csv', index=False)
            pd.Series(feature_names).to_csv('dataset/feature_names.csv', index=False)
            
            # Save metrics
            os.makedirs('models/trained_models', exist_ok=True)
            joblib.dump(metrics, f'models/trained_models/metrics_{selected_model.replace(" ", "_")}.pkl')
            
        st.success(f"Model {selected_model} trained and saved successfully!")
        
        st.subheader("Model Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", f"{metrics['R2']:.4f}")
        col2.metric("MAE", f"{metrics['MAE']:.4f} kPa")
        col3.metric("RMSE", f"{metrics['RMSE']:.4f} kPa")
        
        if metrics['R2'] > 0.95:
            st.balloons()
            st.success("Target R² > 0.95 achieved! The model captures the physics-based behavior exceptionally well.")
