import starlette.middleware.gzip
# Monkeypatch to fix Streamlit/Starlette compatibility issue
if not hasattr(starlette.middleware.gzip, 'DEFAULT_EXCLUDED_CONTENT_TYPES'):
    starlette.middleware.gzip.DEFAULT_EXCLUDED_CONTENT_TYPES = {
        "text/html", "text/css", "text/javascript", "application/javascript",
        "application/json", "application/x-javascript",
    }

import streamlit as st
import os

st.set_page_config(
    page_title="Silo Stress Predictor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏭 Physics-Informed ML for Reinforced Concrete Silo Stress Prediction")

st.markdown("""
### Welcome to the Silo Engineering Analytics Platform
by Professional Bhai's Twati Tivani and Traful
This platform integrates **Janssen’s Silo Theory** with **Machine Learning** to predict hoop stress in reinforced concrete silos.

#### Features:
- **Dataset Generator:** Create synthetic datasets using physics-based equations with optional noise.
- **ML Training:** Train various regression models (Random Forest, XGBoost, Gradient Boosting) to predict hoop stress.
- **Prediction Dashboard:** Input silo parameters to get real-time stress predictions from ML models compared against theoretical equations.
- **Visualization:** Interactive graphs exploring structural behavior, error distribution, and feature importance.
- **Model Comparison:** Evaluate and compare performance metrics (R², MAE, RMSE) of trained models.

#### Start Exploring:
Navigate through the sidebar pages to generate data, train models, and analyze results.
""")

# Ensure necessary directories exist
os.makedirs('dataset', exist_ok=True)
os.makedirs('models/trained_models', exist_ok=True)
os.makedirs('reports', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

st.info("System directories initialized. Please navigate through the sidebar to begin your engineering analysis.")
