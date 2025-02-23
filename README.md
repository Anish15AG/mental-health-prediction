# Mental Health Prediction App

## Overview
The Mental Health Prediction App is a self-analysis tool designed to predict possible mental health conditions based on user-provided symptoms and details. The project combines robust machine learning techniques with a natural language explanation component. Two models—Random Forest and XGBoost—are developed for multi-class classification, while a LLM generates detailed explanations and suggestions regarding the predictions.

This application is built for easy integration into chatbots or web-based interfaces, offering both command-line and interactive UI (Streamlit) options.

## Repository Structure
- **mental_health_ui.py**  
  Streamlit-based UI that allows users to enter patient data, view model predictions, and read generated explanations.

- **predict_mental_health.py**  
  Pre-processing datasets, training and testing the models with comparison as well as saving them. 
  
- **llm_explanations.py**
    Module containing the logic for generating natural language Module generating explanations using Google's Gemini 2.0 via OpenRouter. Leverages API calls to provide mental health insights.


## Getting Started
### Requirements
- Python 3.7 or higher
### Install the required dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
### How to Run the Files
Run the prediction script:
```bash
python src/predict_mental_health.py
```
Launch the Streamlit UI:
```bash
streamlit run src/mental_health_ui.py
```