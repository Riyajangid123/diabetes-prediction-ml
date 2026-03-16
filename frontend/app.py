import streamlit as st
import requests

st.set_page_config(page_title="Diabetes Prediction System", layout="centered")

st.title("Diabetes Prediction System")
st.write("Enter patient details to predict diabetes risk.")

st.markdown("---")

# Input fields
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)

glucose = st.number_input("Glucose Level", min_value=0.0)

blood_pressure = st.number_input("Blood Pressure", min_value=0.0)

skin_thickness = st.number_input("Skin Thickness", min_value=0.0)

insulin = st.number_input("Insulin Level", min_value=0.0)

bmi = st.slider("BMI (Body Mass Index)",min_value=10.0,max_value=60.0,value=25.0,step=0.1)

diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0)

age = st.number_input("Age", min_value=1, max_value=120)

st.markdown("---")

# Predict button
if st.button("Predict Diabetes"):

    input_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=input_data
    )

    result = response.json()

    st.subheader("Prediction Result")
    st.success(result["prediction"])