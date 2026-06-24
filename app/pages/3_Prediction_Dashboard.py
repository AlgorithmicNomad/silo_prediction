import os
import pandas as pd
import streamlit as st
import sys
from pathlib import Path
import joblib
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

try:
    from utils.physics import calculate_hoop_stress, calculate_lateral_pressure
    from utils.training import load_model
except ImportError as e:
    st.error(f"Utility modules could not be loaded: {e}")
    st.stop()


def render_safety_status(stress_value, source_label):
    limit = 2000.0
    stress_ratio = stress_value / limit

    st.subheader(f"Safety Status ({source_label})")
    st.metric("Stress Utilization", f"{stress_ratio * 100:.1f}% of limit")

    if stress_ratio < 0.4:
        st.success("SAFE: Stress is comfortably within the assumed design limit.")
    elif stress_ratio < 0.75:
        st.warning("MODERATE: Stress is rising. Review reinforcement and operating margin.")
    elif stress_ratio <= 1.0:
        st.warning("HIGH RISK: Stress is close to the assumed design limit.")
    else:
        st.error("UNSAFE: Predicted stress exceeds the assumed design limit.")
        st.info("Recommendation: Increase wall thickness or reduce pressure demand.")


def load_model_metrics():
    model_names = ["Random Forest", "Gradient Boosting", "XGBoost"]
    metrics_rows = []

    for model_name in model_names:
        metrics_path = (
            root_dir
            / "models"
            / "trained_models"
            / f"metrics_{model_name.replace(' ', '_')}.pkl"
        )
        if metrics_path.exists():
            metrics = joblib.load(metrics_path)
            metrics_rows.append(
                {
                    "Model": model_name,
                    "R2": metrics.get("R2"),
                    "MAE": metrics.get("MAE"),
                    "RMSE": metrics.get("RMSE"),
                }
            )

    if metrics_rows:
        return pd.DataFrame(metrics_rows)
    return pd.DataFrame()


def get_safety_summary(stress_value):
    limit = 2000.0
    stress_ratio = stress_value / limit

    if stress_ratio < 0.4:
        return "SAFE", "Stress is comfortably within the assumed design limit."
    if stress_ratio < 0.75:
        return "MODERATE", "Stress is rising. Review reinforcement and operating margin."
    if stress_ratio <= 1.0:
        return "HIGH RISK", "Stress is close to the assumed design limit."
    return "UNSAFE", "Predicted stress exceeds the assumed design limit."


def build_pdf_report(
    selected_model_name,
    inputs,
    analytical_p_z,
    analytical_sigma_h,
    ml_pred,
    metrics_df,
):
    buffer = BytesIO()
    analytical_status, analytical_note = get_safety_summary(analytical_sigma_h)
    analytical_utilization = (analytical_sigma_h / 2000.0) * 100

    if ml_pred is not None:
        ml_status, ml_note = get_safety_summary(ml_pred)
        ml_utilization = (ml_pred / 2000.0) * 100
        prediction_error = ml_pred - analytical_sigma_h
    else:
        ml_status, ml_note = "NOT AVAILABLE", "Selected ML model is not trained yet."
        ml_utilization = None
        prediction_error = None

    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        y = 0.97

        def line(text, step=0.03, size=10, weight="normal"):
            nonlocal y
            fig.text(0.08, y, text, fontsize=size, fontweight=weight, family="DejaVu Sans")
            y -= step

        line("Silo Stress Prediction Report", step=0.04, size=18, weight="bold")
        line("Generated from the Prediction Dashboard", step=0.05, size=11)

        line("Input Parameters", step=0.035, size=13, weight="bold")
        for key, value in inputs.items():
            line(f"{key}: {value}", step=0.028)

        y -= 0.01
        line("Analytical Results (Janssen)", step=0.035, size=13, weight="bold")
        line(f"Lateral pressure p(z): {analytical_p_z:.2f} kPa", step=0.028)
        line(f"Hoop stress sigma_h: {analytical_sigma_h:.2f} kPa", step=0.028)
        line(f"Safety status: {analytical_status}", step=0.028)
        line(f"Stress utilization: {analytical_utilization:.1f}% of limit", step=0.028)
        line(f"Comment: {analytical_note}", step=0.04)

        line(f"ML Results ({selected_model_name})", step=0.035, size=13, weight="bold")
        if ml_pred is not None:
            line(f"Predicted hoop stress sigma_h: {ml_pred:.2f} kPa", step=0.028)
            line(f"Prediction error vs analytical: {prediction_error:.2f} kPa", step=0.028)
            line(f"Safety status: {ml_status}", step=0.028)
            line(f"Stress utilization: {ml_utilization:.1f}% of limit", step=0.028)
            line(f"Comment: {ml_note}", step=0.04)
        else:
            line("Prediction unavailable because the selected model is not trained.", step=0.04)

        line("Model Accuracy Summary", step=0.035, size=13, weight="bold")
        if metrics_df.empty:
            line("No trained model metrics found.", step=0.03)
        else:
            for _, row in metrics_df.iterrows():
                line(
                    f"{row['Model']}: R2={row['R2']:.4f}, MAE={row['MAE']:.2f}, RMSE={row['RMSE']:.2f}",
                    step=0.028,
                )

        y -= 0.01
        line("Engineering Interpretation", step=0.035, size=13, weight="bold")
        line(
            f"At depth {inputs['Depth Position from Top (z) [m]']}, the current silo "
            f"configuration produces an analytical hoop stress of {analytical_sigma_h:.2f} kPa.",
            step=0.03,
        )
        if ml_pred is not None:
            line(
                f"The selected ML model predicts {ml_pred:.2f} kPa for the same condition.",
                step=0.03,
            )

        plt.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()


st.set_page_config(page_title="Prediction Dashboard", page_icon="chart_with_upwards_trend")

st.title("Real-Time Prediction Dashboard")

with st.sidebar:
    st.header("Silo Geometric Parameters")
    H = st.slider("Silo Total Height (H) [m]", 10.0, 50.0, 30.0)
    D = st.slider("Silo Diameter (D) [m]", 5.0, 20.0, 10.0)
    t = st.slider("Wall Thickness (t) [m]", 0.1, 0.5, 0.25)

    st.header("Material Properties")
    gamma = st.slider("Bulk Density (gamma) [kN/m^3]", 8.0, 16.0, 12.0)
    mu = st.slider("Wall Friction Coeff (mu)", 0.3, 0.6, 0.45)
    K = st.slider("Lateral Pressure Coeff (K)", 0.4, 0.6, 0.5)

    st.header("Analysis Depth")
    z = st.slider("Depth Position from Top (z) [m]", 0.0, H, H / 2)

    st.header("Model Selection")
    selected_model_name = st.selectbox(
        "Select Trained ML Model",
        ["Random Forest", "Gradient Boosting", "XGBoost"],
    )

analytical_p_z = calculate_lateral_pressure(gamma, D, mu, K, z)
analytical_sigma_h = calculate_hoop_stress(analytical_p_z, D, t)
model, scaler = load_model(selected_model_name)
ml_pred = None

st.header("Engineering Analysis Results")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Analytical (Janssen)")
    st.metric("Lateral Pressure p(z)", f"{analytical_p_z:.2f} kPa")
    st.metric("Hoop Stress sigma_h", f"{analytical_sigma_h:.2f} kPa")
    render_safety_status(analytical_sigma_h, "Analytical")

with col2:
    st.subheader(f"ML Prediction ({selected_model_name})")
    if model is not None and scaler is not None:
        try:
            feature_file = "dataset/feature_names.csv"
            if os.path.exists(feature_file):
                features_list = pd.read_csv(feature_file).iloc[:, 0].tolist()
            else:
                features_list = [
                    "Height_H",
                    "Diameter_D",
                    "Thickness_t",
                    "Density_gamma",
                    "Friction_mu",
                    "Pressure_Coeff_K",
                    "Depth_z",
                ]

            input_df = pd.DataFrame(
                {
                    "Height_H": [H],
                    "Diameter_D": [D],
                    "Thickness_t": [t],
                    "Density_gamma": [gamma],
                    "Friction_mu": [mu],
                    "Pressure_Coeff_K": [K],
                    "Depth_z": [z],
                }
            )
            input_df = input_df[features_list]

            input_scaled = scaler.transform(input_df)
            ml_pred = model.predict(input_scaled)[0]
            error = ml_pred - analytical_sigma_h

            st.metric(
                "Predicted Hoop Stress sigma_h",
                f"{ml_pred:.2f} kPa",
                delta=f"{error:.2f} kPa Error",
                delta_color="inverse",
            )
            render_safety_status(ml_pred, "ML Prediction")
        except Exception as e:
            st.error(f"Error during ML inference: {e}")
    else:
        st.warning(f"Model '{selected_model_name}' not available. Train the model first.")
        st.info("The analytical safety status on the left still updates in real time.")

st.markdown("---")
st.subheader("Physics Interpretation")
st.write(
    f"At depth **{z:.2f} m**, the current inputs produce an analytical hoop stress of "
    f"**{analytical_sigma_h:.2f} kPa**. The safety status above updates immediately "
    f"whenever the user changes geometry, material properties, or depth."
)

metrics_df = load_model_metrics()
st.markdown("---")
st.subheader("Model Accuracy")

if metrics_df.empty:
    st.info("No trained model metrics found yet. Train the models to see their accuracy here.")
else:
    for _, row in metrics_df.iterrows():
        accuracy_pct = row["R2"] * 100 if pd.notna(row["R2"]) else None
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model", row["Model"])
        c2.metric("Accuracy (R2)", f"{accuracy_pct:.2f}%")
        c3.metric("MAE", f"{row['MAE']:.2f}")
        c4.metric("RMSE", f"{row['RMSE']:.2f}")

report_inputs = {
    "Silo Total Height (H) [m]": f"{H:.2f}",
    "Silo Diameter (D) [m]": f"{D:.2f}",
    "Wall Thickness (t) [m]": f"{t:.2f}",
    "Bulk Density (gamma) [kN/m^3]": f"{gamma:.2f}",
    "Wall Friction Coeff (mu)": f"{mu:.2f}",
    "Lateral Pressure Coeff (K)": f"{K:.2f}",
    "Depth Position from Top (z) [m]": f"{z:.2f}",
}

pdf_bytes = build_pdf_report(
    selected_model_name,
    report_inputs,
    analytical_p_z,
    analytical_sigma_h,
    ml_pred,
    metrics_df,
)

st.markdown("---")
st.subheader("Download Report")
st.download_button(
    label="Download Full PDF Report",
    data=pdf_bytes,
    file_name="silo_prediction_report.pdf",
    mime="application/pdf",
)
