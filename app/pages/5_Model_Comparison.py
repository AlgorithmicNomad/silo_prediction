import streamlit as st
import pandas as pd
import os
import joblib
import plotly.express as px

st.set_page_config(page_title="Model Comparison", page_icon="🏅")

st.title("🏅 Machine Learning Model Performance Comparison")

st.markdown("""
Compare the accuracy and efficiency of different regression architectures trained on the Janssen Silo Dataset.
""")

models = ["Random Forest", "Gradient Boosting", "XGBoost"]
metrics_data = []

for model in models:
    metrics_path = f'models/trained_models/metrics_{model.replace(" ", "_")}.pkl'
    if os.path.exists(metrics_path):
        m = joblib.load(metrics_path)
        m['Model'] = model
        metrics_data.append(m)

if not metrics_data:
    st.warning("No performance metrics available. Please train models in the 'ML Training' page.")
else:
    metrics_df = pd.DataFrame(metrics_data).set_index('Model')
    
    st.subheader("Summary Table")
    st.dataframe(metrics_df.style.highlight_max(subset=['R2'], color='lightgreen').highlight_min(subset=['MAE', 'RMSE'], color='lightgreen'))
    
    metrics_df_reset = metrics_df.reset_index()
    
    col1, col2 = st.columns(2)
    with col1:
        fig_r2 = px.bar(metrics_df_reset, x='Model', y='R2', 
                         title="R² Score (Closer to 1.0 is Better)", 
                         color='Model', text_auto='.4f')
        fig_r2.update_layout(yaxis_range=[0.9, 1.0])
        st.plotly_chart(fig_r2, use_container_width=True)
        
    with col2:
        fig_rmse = px.bar(metrics_df_reset, x='Model', y='RMSE', 
                          title="RMSE (Lower is Better)", 
                          color='Model', text_auto='.2f')
        st.plotly_chart(fig_rmse, use_container_width=True)
        
    st.info("💡 **Insight:** Tree-based ensemble models typically excel at capturing the non-linear exponential relationships present in Janssen's silo formulas.")
