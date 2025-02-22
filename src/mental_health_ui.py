import streamlit as st
import pandas as pd
import joblib
import os


@st.cache_resource
def load_artifacts():
    """Load saved models, scaler, and selected features."""
    rf_model = joblib.load(os.path.join("models/random_forest_model.pkl"))
    xgb_model = joblib.load(os.path.join("models/xgboost_model.pkl"))
    scaler = joblib.load(os.path.join("models/scaler.pkl"))
    selected_features = joblib.load(os.path.join("models/selected_features.pkl"))
    return rf_model, xgb_model, scaler, selected_features

def predict(user_input_df):
    """Make predictions with both models on the provided user input."""
    rf_model, xgb_model, scaler, selected_features = load_artifacts()
    
    # Ensure the input DataFrame contains the expected features.
    df_selected = user_input_df[selected_features]
    input_scaled = scaler.transform(df_selected)
    
    rf_pred = rf_model.predict(input_scaled)[0]
    xgb_pred = xgb_model.predict(input_scaled)[0]
    
    return {"Random Forest Prediction": int(rf_pred), "XGBoost Prediction": int(xgb_pred)}