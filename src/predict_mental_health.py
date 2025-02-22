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

# Fill missing numeric values with the median and categorical values with the mode.
data.fillna(data.median(numeric_only=True), inplace=True)
for col in data.select_dtypes(include=['object']).columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Encode all categorical variables using LabelEncoder.
# This converts text labels into numerical values.
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# 2. Define an Evaluation Function
###############################################
def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a model on test data and print overall performance metrics.
    
    Parameters:
      model: The trained model to be evaluated.
      X_test: Test feature matrix.
      y_test: Test target vector.
      model_name: Name of the model for display.
    
    Returns:
      A dictionary containing Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    try:
        roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    except Exception:
        roc = np.nan
    # Print detailed model performance
    print(f"--- {model_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc}")
    print(classification_report(y_test, y_pred, zero_division=0))
    return {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1, 'ROC-AUC': roc}


#D PIPELINE: Target = 'suicidal'
###############################################
# This section implements the original pipeline using 'suicidal' as the target.

# Separate features and target. Remove the 'suicidal' column and 'id' from features.
X_old = data.drop(columns=['suicidal', 'id'], errors='ignore')
y_old = data['suicidal']

# Use a Random Forest to determine feature importance and select the top 10 features.
rf_fs_old = RandomForestClassifier(random_state=42)
rf_fs_old.fit(X_old, y_old)
fi_old = pd.DataFrame({'Feature': X_old.columns, 'Importance': rf_fs_old.feature_importances_})
fi_old.sort_values(by='Importance', ascending=False, inplace=True)
top_features_old = fi_old["Feature"].values[:10]
X_old = X_old[top_features_old]

# Split the data into training and testing sets.
X_train_old, X_test_old, y_train_old, y_test_old = train_test_split(
    X_old, y_old, test_size=0.2, random_state=42
)

# Standardize features using StandardScaler.
scaler_old = StandardScaler()
X_train_old = scaler_old.fit_transform(X_train_old)
X_test_old = scaler_old.transform(X_test_old)