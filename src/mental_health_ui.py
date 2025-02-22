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