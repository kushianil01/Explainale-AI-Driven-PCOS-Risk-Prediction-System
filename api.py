from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle

from pcos_app import (
    categorize_risk,
    confidence_level,
    generate_recommendations
)

# -----------------------------------
# Initialize FastAPI app
# -----------------------------------

app = FastAPI()

# -----------------------------------
# CORS (IMPORTANT for frontend)
# -----------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change later for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------
# Load pre-trained model
# -----------------------------------

with open("pcos_xgb_trained.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------------
# Define feature columns (IMPORTANT)
# -----------------------------------

feature_cols = [
    "age",
    "height_cm",
    "weight_kg",
    "bmi",
    "cycle_length_days",
    "cycle_length_variation",
    "cycle_irregular_flag",
    "hair_loss",
    "pimples",
    "skin_darkening",
    "hair_growth",
    "symptom_score",
    "exercise_level",
    "diet_quality",
    "stress_score",
    "sleep_hours",
    "bmi_symptom_interaction",
    "cycle_stress_interaction"
]

# -----------------------------------
# Health check route
# -----------------------------------

@app.get("/")
def home():
    return {"message": "PCOS API is running"}

# -----------------------------------
# Prediction endpoint
# -----------------------------------

@app.post("/predict")
def predict(data: dict):

    # -----------------------------
    # Extract input values
    # -----------------------------

    age = data["age"]
    height_cm = data["height_cm"]
    weight_kg = data["weight_kg"]

    cycle_length_days = data["cycle_length_days"]
    cycle_length_variation = data["cycle_length_variation"]

    hair_loss = data["hair_loss"]
    pimples = data["pimples"]
    skin_darkening = data["skin_darkening"]
    hair_growth = data["hair_growth"]

    exercise_level = data["exercise_level"]
    diet_quality = data["diet_quality"]
    stress_score = data["stress_score"]
    sleep_hours = data["sleep_hours"]

    # -----------------------------
    # Derived features
    # -----------------------------

    bmi = weight_kg / ((height_cm / 100) ** 2)

    symptom_score = (
        hair_loss +
        pimples +
        skin_darkening +
        hair_growth
    )

    cycle_irregular_flag = int(
        (cycle_length_days > 35) or
        (cycle_length_days < 24) or
        (cycle_length_variation > 7)
    )

    bmi_symptom_interaction = bmi * symptom_score
    cycle_stress_interaction = cycle_length_days * stress_score

    # -----------------------------
    # Build dataframe
    # -----------------------------

    input_dict = {
        "age": age,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "bmi": bmi,
        "cycle_length_days": cycle_length_days,
        "cycle_length_variation": cycle_length_variation,
        "cycle_irregular_flag": cycle_irregular_flag,
        "hair_loss": hair_loss,
        "pimples": pimples,
        "skin_darkening": skin_darkening,
        "hair_growth": hair_growth,
        "symptom_score": symptom_score,
        "exercise_level": exercise_level,
        "diet_quality": diet_quality,
        "stress_score": stress_score,
        "sleep_hours": sleep_hours,
        "bmi_symptom_interaction": bmi_symptom_interaction,
        "cycle_stress_interaction": cycle_stress_interaction
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_cols]

    # -----------------------------
    # Prediction
    # -----------------------------

    prob = model.predict_proba(input_df.values)[0, 1]
    risk_level = categorize_risk(prob)
    confidence = confidence_level(prob)

    # -----------------------------
    # Recommendations
    # -----------------------------

    recs = generate_recommendations(
        age=age,
        bmi=bmi,
        cycle_length_days=cycle_length_days,
        cycle_irregular_flag=cycle_irregular_flag,
        symptom_score=symptom_score,
        hair_loss=hair_loss,
        pimples=pimples,
        skin_darkening=skin_darkening,
        hair_growth=hair_growth,
        exercise_level=exercise_level,
        diet_quality=diet_quality,
        stress_score=stress_score,
        sleep_hours=sleep_hours,
        risk_level=risk_level
    )

    # -----------------------------
    # Response
    # -----------------------------

    return {
        "probability": float(prob),
        "risk_level": risk_level,
        "confidence": confidence,
        "bmi": round(bmi, 1),
        "recommendations": recs
    }
