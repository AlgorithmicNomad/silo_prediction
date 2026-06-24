# Physics-Informed ML for Concrete Silo Stress Prediction

## 🚀 Project Overview
This project is a Streamlit-based engineering analytics platform for predicting hoop stress in reinforced concrete silos using a physics-informed machine learning pipeline.

It combines:
- **Janssen’s Silo Theory** for analytical lateral pressure and hoop stress calculation,
- **Synthetic dataset generation** from physics-based equations,
- **ML model training** with Random Forest, Gradient Boosting, and XGBoost,
- **Interactive dashboards** for prediction, visualization, and model comparison.

## 🧩 Key Features
- Generate physics-based synthetic silo datasets with optional noise.
- Train regression models on the generated data.
- Compare model performance using R², MAE, and RMSE metrics.
- View analytical vs ML predictions in a real-time dashboard.
- Download PDF engineering reports from the prediction page.

## ⚡ Quick Start
1. Activate the project virtual environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
2. Generate data and train models:
   ```powershell
   python generate_artifacts.py
   ```
3. Run the Streamlit app:
   ```powershell
   python -m streamlit run app.py
   ```
4. Open the browser link shown by Streamlit and use the sidebar pages.

## 🧪 What Works
- `generate_artifacts.py` runs successfully and generates the dataset plus trained XGBoost and Random Forest models.
- The app uses a Streamlit multi-page interface in `app/pages/`.
- The following pages are available:
  - Dataset Generator
  - ML Training
  - Prediction Dashboard
  - Visualization Dashboard
  - Model Comparison
  - About Project

## 📂 Project Structure
- `app/`: Streamlit app entrypoint and page modules
- `dataset/`: Generated dataset + evaluation/prediction CSVs
- `models/`: Trained model artifacts and metrics
- `utils/`: Physics, preprocessing, and training utility modules
- `generate_artifacts.py`: Bootstrap script to create data and train base models

## ⚙️ Installation and Run Instructions
1. Open PowerShell in the project root.
2. Create and activate the virtual environment:
   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
4. Generate dataset and train models:
   ```powershell
   python generate_artifacts.py
   ```
5. Start the Streamlit dashboard:
   ```powershell
   python -m streamlit run app.py
   ```

## 📌 Notes
- Make sure the `.venv` virtual environment is activated when running the app.
- If `XGBoost` fails to import, reinstall dependencies inside `.venv`.
- The app expects generated data in `dataset/generated_dataset.csv` and trained models in `models/trained_models/`.

## 📄 License
This project is licensed under the MIT License. See `LICENSE` for details.
