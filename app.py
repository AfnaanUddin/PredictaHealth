import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit UI
st.title("🩺 PredictaHealth: Diabetes Prediction App")

# Input fields
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=6)
glucose = st.number_input("Glucose Concentration", min_value=0, max_value=200, value=200)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=35)
insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=850, value=125)
bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=50.0, value=32.5)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.0, value=0.5)
age = st.number_input("Age", min_value=21, max_value=100, value=45)

# Prepare input data as a DataFrame
input_data = {
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [diabetes_pedigree_function],
    'Age': [age]
}
input_df = pd.DataFrame(input_data)

# Standardize the input data
input_data_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_data_scaled)

# Output
if prediction[0] == 0:
    st.success("✅ The person is **not diabetic**.")
else:
    st.error("⚠️ The person is **diabetic**.")
