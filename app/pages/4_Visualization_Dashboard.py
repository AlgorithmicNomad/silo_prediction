import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

try:
    from utils.training import load_model
except ImportError as e:
    st.error(f"Utility modules could not be loaded: {e}")
    st.stop()

st.set_page_config(page_title="Visualization Dashboard", page_icon="📊", layout="wide")

st.title("📊 Engineering Visualization Dashboard")

@st.cache_data
def load_full_data():
    path = 'dataset/generated_dataset.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

df = load_full_data()

if df is None:
    st.warning("No dataset found. Please generate data in the 'Dataset Generator' page.")
else:
    st.sidebar.header("Visualization Settings")
    viz_type = st.sidebar.selectbox("Select Graph Type", 
        ["Depth vs Stress Profile", "Actual vs Predicted", "Feature Importance", "Error Distribution", "3D Stress Surface", "Correlation Heatmap"]
    )
    
    if viz_type == "Depth vs Stress Profile":
        st.subheader("Hoop Stress Distribution with Depth")
        # Sample for performance
        df_sample = df.sample(min(2000, len(df)))
        fig = px.scatter(df_sample, x="Depth_z", y="Hoop_Stress_sigma_h", color="Lateral_Pressure_p",
                         title="Variation of Hoop Stress along Silo Height",
                         labels={"Depth_z": "Depth from Top (m)", "Hoop_Stress_sigma_h": "Hoop Stress (kPa)"},
                         color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Correlation Heatmap":
        st.subheader("Engineering Parameter Correlation")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap="RdYlBu", fmt=".2f", ax=ax)
        st.pyplot(fig)

    elif viz_type == "3D Stress Surface":
        st.subheader("3D Interactive Stress Analysis")
        df_sample = df.sample(min(3000, len(df)))
        fig = px.scatter_3d(df_sample, x='Depth_z', y='Thickness_t', z='Hoop_Stress_sigma_h',
                            color='Hoop_Stress_sigma_h',
                            title="Hoop Stress vs Depth & Wall Thickness",
                            labels={'Depth_z': 'Depth (m)', 'Thickness_t': 'Thickness (m)', 'Hoop_Stress_sigma_h': 'Stress (kPa)'},
                            opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)

    else:
        # These visualizations require a trained model
        model_name = st.sidebar.selectbox("Analyze Model", ["Random Forest", "Gradient Boosting", "XGBoost"])
        pred_path = f'dataset/predictions_{model_name.replace(" ", "_")}.csv'
        evaluation_path = f'dataset/evaluation_{model_name.replace(" ", "_")}.csv'
        
        if os.path.exists(evaluation_path):
            evaluation_df = pd.read_csv(evaluation_path)
            y_test = evaluation_df['Actual']
            y_pred = evaluation_df['Predicted']
        elif os.path.exists('dataset/y_test.csv') and os.path.exists(pred_path):
            y_test = pd.read_csv('dataset/y_test.csv')['Actual']
            y_pred = pd.read_csv(pred_path)['Predicted']
            if len(y_test) != len(y_pred):
                st.error(
                    f"Saved evaluation files are out of sync for {model_name}: "
                    f"Actual has {len(y_test)} rows and Predicted has {len(y_pred)} rows. "
                    "Retrain this model to regenerate matched evaluation data."
                )
                st.stop()
        else:
            st.info(f"Model performance data for {model_name} not found. Please train this model first.")
            st.stop()
            
        if viz_type == "Actual vs Predicted":
            st.subheader(f"Model Accuracy Check: {model_name}")
            fig = px.scatter(x=y_test, y=y_pred, opacity=0.4,
                             labels={'x': 'Analytical Stress (kPa)', 'y': 'ML Predicted Stress (kPa)'},
                             title="Predicted vs Analytical Stress (Ideal: y=x)")
            fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
                          line=dict(color="red", dash="dash"))
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Error Distribution":
            st.subheader(f"Residual Analysis: {model_name}")
            errors = y_test - y_pred
            fig = px.histogram(errors, nbins=50, title="Distribution of Prediction Residuals",
                               labels={'value': 'Error (kPa)'}, color_discrete_sequence=['indianred'])
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Feature Importance":
            st.subheader(f"Model Logic: Feature Importances ({model_name})")
            model, _ = load_model(model_name)
            if model:
                if hasattr(model, 'feature_importances_'):
                    features_path = 'dataset/feature_names.csv'
                    if os.path.exists(features_path):
                        features = pd.read_csv(features_path).iloc[:, 0].tolist()
                        importances = model.feature_importances_
                        feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=True)
                        fig = px.bar(feat_imp, x='Importance', y='Feature', orientation='h',
                                     title=f"Relative Importance of Input Parameters", color='Importance')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Feature names file missing.")
                else:
                    st.warning(f"Feature importance not supported by {model_name} architecture.")
            else:
                st.error(f"Failed to load trained model: {model_name}")
