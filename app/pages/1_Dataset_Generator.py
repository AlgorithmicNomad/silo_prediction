import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from utils.physics import generate_synthetic_data

st.set_page_config(page_title="Dataset Generator", page_icon="🗄️")

st.title("🗄️ Physics-Based Dataset Generator")

st.markdown("""
Generate synthetic datasets based on **Janssen’s Silo Theory** equations for lateral pressure and hoop stress.
""")

with st.sidebar:
    st.header("Generator Parameters")
    num_samples = st.number_input("Number of Samples", min_value=1000, max_value=100000, value=10000, step=1000)
    add_noise = st.checkbox("Add Gaussian Noise (1%)", value=True, help="Simulate real-world measurement uncertainties")

if st.button("Generate Dataset", type="primary"):
    with st.spinner("Generating physics-based synthetic data..."):
        df = generate_synthetic_data(num_samples=num_samples, add_noise=add_noise)
        
        # Save to CSV
        os.makedirs('dataset', exist_ok=True)
        csv_path = 'dataset/generated_dataset.csv'
        df.to_csv(csv_path, index=False)
        
        st.success(f"Successfully generated {num_samples} records and saved to `{csv_path}`")
        
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        st.subheader("Dataset Statistics")
        st.dataframe(df.describe())
        
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Dataset as CSV",
            data=csv_data,
            file_name="silo_dataset.csv",
            mime="text/csv"
        )
