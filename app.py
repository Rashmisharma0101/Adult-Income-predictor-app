from preprocessing import AgeBinner, CountryGrouper, CapitalPresence
import streamlit as st
import joblib
import pandas as pd


st.title("Income Prediction app")

model = joblib.load("model.pkl")

age = st.number_input("Age", min_value=0, max_value=100, value=30)
workclass = st.selectbox("Workclass", [
    'Private','Self-emp-not-inc','Self-emp-inc','Federal-gov','Local-gov',
    'State-gov','Without-pay','Never-worked'
])
education_num = st.number_input("Education Num", min_value=1, max_value=16, value=10)
marital_status = st.selectbox("Marital Status", [
    "Never-married", "Divorced", "Separated", "Widowed",
    "Married-civ-spouse", "Married-spouse-absent", "Married-AF-spouse"
])
occupation = st.selectbox("Occupation", [
    "Tech-support","Craft-repair","Other-service","Sales","Exec-managerial",
    "Prof-specialty","Handlers-cleaners","Machine-op-inspct","Adm-clerical",
    "Farming-fishing","Transport-moving","Priv-house-serv","Protective-serv",
    "Armed-Forces"
])
relationship = st.selectbox("Relationship", [
    "Husband","Not-in-family","Own-child","Unmarried","Wife","Other-relative"
])
race = st.selectbox("Race", [
    "White","Black","Asian-Pac-Islander","Amer-Indian-Eskimo","Other"
])
sex = st.radio("Sex", ["Male", "Female"])
capital_gain = st.number_input("Capital Gain", min_value=0)
capital_loss = st.number_input("Capital Loss", min_value=0)
hours_per_week = st.number_input("Hours per Week", min_value=0, max_value=100)
native_country = st.selectbox("Native Country", [
    "United-States","India","Canada","Mexico","Philippines","Germany","Other"
])

input_df = pd.DataFrame([{
    "age": age,
    "workclass": workclass,
    "education-num": education_num,
    "marital-status": marital_status,
    "occupation": occupation,
    "relationship": relationship,
    "race": race,
    "sex": sex,
    "capital-gain": capital_gain,
    "capital-loss": capital_loss,
    "hours-per-week": hours_per_week,
    "native-country": native_country
}])

if st.button("Predict Income"):
    pred = model.predict(input_df)[0]
    st.success(f"Predicted Income Category: {'>50K' if pred == 1 else '<=50K'}")