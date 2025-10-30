# streamlit_app.py
import streamlit as st
import joblib
import numpy as np
from pathlib import Path

# Page setup
st.set_page_config(page_title="Starbucks MLR Prediction Model", layout="centered")

# Title and description
st.title("☕ Starbucks MLR Prediction Model")
st.write("Enter the details below to predict the **number of Cups of Coffee per day**.")

# Path to the saved model
MODEL_PATH = Path("Starbucks_MLR.pkl")

# Load the model safely
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file not found at `{MODEL_PATH}`. Please make sure the file is in the app directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Input fields
st.header("Input Features")
prepaid_amount = st.number_input("Prepaid Card Amount ($)", min_value=0.0, value=25.0, step=1.0)
age = st.number_input("Customer Age (years)", min_value=0, value=30, step=1)
days_per_month = st.number_input("Visits per Month", min_value=0, value=12, step=1)
income = st.number_input("Monthly Income ($)", min_value=0.0, value=35.0, step=1.0)

# Predict button
if st.button("Predict Cups of Coffee per Day"):
    try:
        # Prepare input data
        X = np.array([[prepaid_amount, age, days_per_month, income]], dtype=float)
        
        # Make prediction
        prediction = model.predict(X)
        result = float(prediction[0])
        
        # Display result
        st.success(f"✅ Predicted Cups of Coffee per Day: **{result:.2f}**")
    except Exception as e:
        st.error(f"Prediction error: {e}")
