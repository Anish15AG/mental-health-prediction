import streamlit as st
import pandas as pd
import joblib
import os
from llm_explanations import generate_explanation

@st.cache_resource
def load_artifacts():
    rf_model = joblib.load(os.path.join("models/random_forest_model.pkl"))
    xgb_model = joblib.load(os.path.join("models/xgboost_model.pkl"))
    scaler = joblib.load(os.path.join("models/scaler.pkl"))
    selected_features = joblib.load(os.path.join("models/selected_features.pkl"))
    return rf_model, xgb_model, scaler, selected_features

def predict(user_input_df):
    rf_model, xgb_model, scaler, selected_features = load_artifacts()
    df_selected = user_input_df[selected_features]
    input_scaled = scaler.transform(df_selected)
    rf_pred = rf_model.predict(input_scaled)[0]
    xgb_pred = xgb_model.predict(input_scaled)[0]
    return {"Random Forest Prediction": int(rf_pred), "XGBoost Prediction": int(xgb_pred)}

# --- Streamlit UI ---
st.title("Mental Health Prediction App")
st.markdown("Please enter the following information to receive a mental health assessment.")

# --- PHQ-9 Score ---
st.subheader("PHQ-9 Score (Depression)")
st.markdown("""
The PHQ-9 is a tool used to screen for and measure the severity of depression. 
Scores range from 0 to 27.
""")
st.markdown("""
**Score Interpretation:**
* 0-4: Minimal depression
* 5-9: Mild depression
* 10-14: Moderate depression
* 15-19: Moderately severe depression
* 20-27: Severe depression
""")
phq_score = st.number_input("Enter PHQ-9 Score", min_value=0.0, max_value=27.0, value=12.0)

# --- GAD-7 Score ---
st.subheader("GAD-7 Score (Anxiety)")
st.markdown("""
The GAD-7 is a tool used to screen for and measure the severity of generalized anxiety disorder. 
Scores range from 0 to 21.
""")
st.markdown("""
**Score Interpretation:**
* 0-4: Minimal anxiety
* 5-9: Mild anxiety
* 10-14: Moderate anxiety
* 15-21: Severe anxiety
""")
gad_score = st.number_input("Enter GAD-7 Score", min_value=0, max_value=21, value=15)

# --- Other Numeric Inputs ---
st.subheader("Other Details")
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=22.5)
epworth_score = st.number_input("Epworth Sleepiness Scale Score", min_value=0, max_value=24, value=10)
age = st.number_input("Age", min_value=10, max_value=100, value=25)
school_year = st.number_input("School Year", min_value=1, max_value=10, value=3)
depression_severity = st.number_input("Depression Severity (numeric)", min_value=0.0, max_value=10.0, value=3.0)

# --- Categorical Inputs ---
st.subheader("Categorical Information")
depressiveness = st.selectbox("Depressiveness", options=["True", "False"])
anxiety_severity = st.selectbox("Anxiety Severity", options=["None-minimal", "Mild", "Moderate", "Severe"])
who_bmi = st.selectbox("WHO BMI Category", options=["Normal", "Underweight", "Overweight", "Class I Obesity", "Class II Obesity", "Class III Obesity", "Not Available"])

# --- Mapping Categorical to Numeric ---
depressiveness_numeric = 1 if depressiveness == "True" else 0
anxiety_mapping = {"None-minimal": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
anxiety_severity_numeric = anxiety_mapping[anxiety_severity]
who_bmi_mapping = {"Normal": 0, "Underweight": 1, "Overweight": 2, "Class I Obesity": 3, "Class II Obesity": 4, "Class III Obesity": 5, "Not Available": 6}
who_bmi_numeric = who_bmi_mapping[who_bmi]

# --- User Input DataFrame ---
user_input = {
    "phq_score": phq_score,
    "depressiveness": depressiveness_numeric,
    "bmi": bmi,
    "epworth_score": epworth_score,
    "gad_score": gad_score,
    "depression_severity": depression_severity,
    "age": age,
    "school_year": school_year,
    "anxiety_severity": anxiety_severity_numeric,
    "who_bmi": who_bmi_numeric
}
user_input_df = pd.DataFrame(user_input, index=[0])

# --- Prediction and Explanation ---
if st.button("Get Mental Health Assessment"):
    predictions = predict(user_input_df)
    st.subheader("Assessment Results")
    st.write(predictions)
    explanation = generate_explanation(user_input_df, predictions)
    st.subheader("Explanation and Recommendations")
    st.write(explanation)