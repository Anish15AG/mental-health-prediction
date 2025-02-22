import os
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
import shap
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# 1. Common Data Loading and Preprocessing
###############################################
# Read the CSV file into a DataFrame.
data_path = os.path.join("data", "raw", "depression_anxiety_data.csv")
data = pd.read_csv(data_path)

# Remove duplicate rows to ensure data quality.
data.drop_duplicates(inplace=True)