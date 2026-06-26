"""
Diabetes Risk Assessment Predictor - FastAPI Backend REST API

This script implements a lightweight, high-performance REST API to score diabetes risk profiles.
It loads the trained SVC (Support Vector Classifier) and StandardScaler, scales incoming health data
(imputing missing features with real dataset medians), and outputs risk predictions.
"""

import os
import sys
import joblib
import numpy as np
import warnings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Suppress sklearn feature names warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Support UTF-8 emoji printing on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Risk Assessment API",
    description="Real-time health profiling using a trained SVM model on Pima Indians Diabetes dataset",
    version="1.0.0",
)

# Enable CORS middleware so local HTML frontend can securely interact with it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths relative to the project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "svm_model.pkl")
SCALER_PATH = os.path.join(ROOT_DIR, "models", "scaler.pkl")

# Load model and scaler
try:
    print("Loading model and scaler assets...")
    svm_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and scaler assets loaded successfully.")
except Exception as e:
    print(f"❌ Error loading assets: {e}")
    sys.exit(1)


# Pydantic input schema containing the client-side inputs
class DiabetesInput(BaseModel):
    glucose: float
    bmi: float
    age: int


@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Welcome to the Diabetes Risk Assessment API. Use POST /api/predict to analyze risk profiles.",
    }


@app.post("/api/predict")
def predict_diabetes(input_data: DiabetesInput):
    try:
        raw_glucose = input_data.glucose
        raw_bmi = input_data.bmi
        raw_age = input_data.age

        # Impute missing values with dataset medians:
        # Pregnancies = 3.0, BloodPressure = 72.0, SkinThickness = 23.0, Insulin = 30.5, DiabetesPedigreeFunction = 0.3725
        raw_pregnancies = 3.0
        raw_blood_pressure = 72.0
        raw_skin_thickness = 23.0
        raw_insulin = 30.5
        raw_pedigree = 0.3725

        # 1. Construct 1x8 feature vector in correct column order:
        # ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        features = np.array([[
            raw_pregnancies,
            raw_glucose,
            raw_blood_pressure,
            raw_skin_thickness,
            raw_insulin,
            raw_bmi,
            raw_pedigree,
            raw_age
        ]])

        # 2. Scale features
        scaled_features = scaler.transform(features)
        
        scaled_glucose = float(scaled_features[0][1])
        scaled_bmi = float(scaled_features[0][5])
        scaled_age = float(scaled_features[0][7])

        # 3. Run prediction and probability estimation
        probabilities = svm_model.predict_proba(scaled_features)
        probability = float(probabilities[0][1])

        # 4. Apply Enterprise Risk Threshold
        if probability >= 0.35:
            prediction = 1
            message = "🚨 High Risk Profile Detected"
        else:
            prediction = 0
            message = "✅ Optimal Health Indicators Verified"

        return {
            "prediction": prediction,
            "probability": probability,
            "message": message,
            "details": {
                "raw_glucose": raw_glucose,
                "raw_bmi": raw_bmi,
                "raw_age": raw_age,
                "scaled_glucose": scaled_glucose,
                "scaled_bmi": scaled_bmi,
                "scaled_age": scaled_age,
            },
        }

    except Exception as e:
        return {
            "prediction": -1,
            "probability": 0.0,
            "message": f"An error occurred during scoring: {str(e)}",
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
