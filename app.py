import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def load_artifacts():
    model_path = ARTIFACTS_DIR / "xgboost_demand_model.pkl"
    encoders_path = ARTIFACTS_DIR / "label_encoders.pkl"
    if not model_path.is_file() or model_path.stat().st_size == 0 or not encoders_path.is_file():
        return None
    with model_path.open("rb") as model_file, encoders_path.open("rb") as encoders_file:
        return pickle.load(model_file), pickle.load(encoders_file)


st.title("Demand Forecasting")
artifacts = load_artifacts()
if artifacts is None:
    st.error("Model files are missing. Run all cells in notebooks/machine_learning.ipynb to create them.")
    st.stop()

model, label_encoders = artifacts
st.header("Input Features")
price = st.number_input("Price", min_value=0.0, value=50.0)
discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=10.0)
inventory_level = st.number_input("Inventory Level", min_value=0, value=100)
promotion = st.selectbox("Promotion", [0, 1])
competitor_pricing = st.number_input("Competitor Pricing", min_value=0.0, value=50.0)
category = st.selectbox("Category", label_encoders["Category"].classes_.tolist())

input_data = pd.DataFrame({
    "Price": [price],
    "Discount": [discount],
    "Inventory Level": [inventory_level],
    "Promotion": [promotion],
    "Competitor Pricing": [competitor_pricing],
    "Category": [category],
})
input_data["Category"] = label_encoders["Category"].transform(input_data["Category"])

if st.button("Predict Demand"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Demand: {int(prediction[0])} Units")
