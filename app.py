import streamlit as st
from pathlib import Path
import sys
import os

# Robust path handling
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

# Ensure necessary directories exist
try:
    (BASE_DIR / "dataset").mkdir(exist_ok=True)
    (BASE_DIR / "models" / "trained_models").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "reports").mkdir(exist_ok=True)
    (BASE_DIR / "visualizations").mkdir(exist_ok=True)
except Exception as e:
    st.error(f"Startup error: Could not create necessary directories. Details: {e}")

st.set_page_config(
    page_title="Silo Stress Predictor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏭 Physics-Informed ML for Reinforced Concrete Silo Stress Prediction ")

st.markdown("""
### Welcome to the Silo Engineering Analytics Platform -by Professional Bhai's Twati Tivani and Traful
This platform integrates **Janssen’s Silo Theory** with **Machine Learning** to predict hoop stress in reinforced concrete silos.
by Professional Bhai's Twati Tivani and Traful
#### Features:
- **Dataset Generator:** Create synthetic datasets using physics-based equations with optional noise.
- **ML Training:** Train various regression models (Random Forest, XGBoost, Gradient Boosting) to predict hoop stress.
- **Prediction Dashboard:** Input silo parameters to get real-time stress predictions from ML models compared against theoretical equations.
- **Visualization:** Interactive graphs exploring structural behavior, error distribution, and feature importance.
- **Model Comparison:** Evaluate and compare performance metrics (R², MAE, RMSE) of trained models.

#### Start Exploring:
Navigate through the sidebar pages to generate data, train models, and analyze results.
""")

dataset_path = BASE_DIR / "dataset" / "generated_dataset.csv"
if not dataset_path.exists():
    st.info("👋 No dataset found. Please navigate to the **Dataset Generator** page to create one.")
else:
    st.success("✅ Dataset found and ready for analysis.")
    df_preview = (BASE_DIR / "dataset" / "generated_dataset.csv")
    try:
        import pandas as pd
        df = pd.read_csv(df_preview, nrows=5)
        st.subheader("Dataset Preview")
        st.dataframe(df)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

st.sidebar.info("Select a page above to begin.")
