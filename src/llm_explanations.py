import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

def get_phq_severity(score):
    if score <= 4:
        return "Minimal"
    elif 5 <= score <= 9:
        return "Mild"
    elif 10 <= score <= 14:
        return "Moderate"
    elif 15 <= score <= 19:
        return "Moderately Severe"
    else:
        return "Severe"

def get_gad_severity(score):
    if score <= 4:
        return "Minimal"
    elif 5 <= score <= 9:
        return "Mild"
    elif 10 <= score <= 14:
        return "Moderate"
    elif 15 <= score <= 21:
        return "Severe"

def generate_explanation(user_input, predictions):
    try:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        site_url = ""
        site_name = ""

        phq_score = user_input["phq_score"].values[0]
        gad_score = user_input["gad_score"].values[0]
        age = user_input["age"].values[0]

        rf_prediction = predictions.get("Random Forest Prediction", "N/A")
        xgb_prediction = predictions.get("XGBoost Prediction", "N/A")

        phq_severity = get_phq_severity(phq_score)
        gad_severity = get_gad_severity(gad_score)

        prompt = (
            f"Patient Profile:\n"
            f"  - Age: {age}\n"
            f"  - PHQ Score (Depression): {phq_score} (Severity: {phq_severity})\n"
            f"  - GAD Score (Anxiety): {gad_score} (Severity: {gad_severity})\n"
            f"Model Predictions:\n"
            f"  - Random Forest: {rf_prediction}\n"
            f"  - XGBoost: {xgb_prediction}\n"
            #f"Based on this patient profile, especially the PHQ score of {phq_score} (Severity: {phq_severity}) and GAD score of {gad_score} (Severity: {gad_severity}), "
            f"Please provide a very detailed and elaborate explanation of the model predictions. "
            f"Pay close attention to the PHQ and GAD scores and highlight any discrepancies. "
            f"Suggest at least 5-6 detailed coping mechanisms with specific instructions on how to implement them. "
            f"Provide comprehensive advice on the next steps for seeking professional mental health support. "
            f"Use bullet points and numbered lists to structure your response for clarity. "
            f"Provide a very long and detailed response."
        )

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": site_url,
                "X-Title": site_name,
            },
            data=json.dumps({
                "model": "google/gemini-2.0-flash-thinking-exp:free",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            })
        )

        response.raise_for_status()

        response_json = response.json()
        generated_text = response_json["choices"][0]["message"]["content"]
        return generated_text.strip()

    except requests.exceptions.RequestException as e:
        return f"Error during API request: {e}"
    except KeyError as e:
        return f"Error parsing API response: Missing key {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"