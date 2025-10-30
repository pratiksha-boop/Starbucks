# streamlit_app.py
import streamlit as st
import joblib
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Starbucks MLR prediction Model", layout="centered")

st.title("Starbucks MLR Prediction Model")
st.write("Provide the following values to predict the number of Cups of Coffee per day.")

MODEL_PATH = Path("Starbucks_MLR.pkl")

# Load model safely
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model from '{MODEL_PATH}': {e}")
    st.stop()

# Inputs
prepaid_amount = st.number_input("Amount of Prepaid card", min_value=0.0, value=25.0, step=1.0)
age = st.number_input("Age", min_value=0, value=30, step=1)
days_per_month = st.number_input("Days per month at Starbucks", min_value=0, value=12, step=1)
income = st.number_input("Income", min_value=0.0, value=35.0, step=1.0)

if st.button("Predict"):
    try:
        X = np.array([[prepaid_amount, age, days_per_month, income]], dtype=float)
        prediction = model.predict(X)
        result = float(prediction[0])
        st.success(f"☕ Predicted Cups of Coffee per day: **{result:.2f}**")
    except Exception as e:
        st.error(f"Prediction error: {e}")
