from fastapi import FastAPI
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

from pcos_app import (
    train_pcos_model,
    categorize_risk,
    confidence_level,
    generate_recommendations
)

app = FastAPI()
# ✅ ADD THIS RIGHT AFTER app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model, feature_cols, _, _ = train_pcos_model()

@app.post("/predict")
def predict(data: dict):
    # Convert input
    input_df = pd.DataFrame([data])

    # Calculate BMI
    bmi = data["weight_kg"] / ((data["height_cm"] / 100) ** 2)

    # Derived fields
    symptom_score = (
        data["hair_loss"] +
        data["pimples"] +
        data["skin_darkening"] +
        data["hair_growth"]
    )

    cycle_irregular_flag = int(
        (data["cycle_length_days"] > 35) or
        (data["cycle_length_days"] < 24) or
        (data["cycle_length_variation"] > 7)
    )

    input_df["bmi"] = bmi
    input_df["symptom_score"] = symptom_score
    input_df["cycle_irregular_flag"] = cycle_irregular_flag
    input_df["bmi_symptom_interaction"] = bmi * symptom_score
    input_df["cycle_stress_interaction"] = data["cycle_length_days"] * data["stress_score"]

    # Align columns
    for col in feature_cols:
        if col not in input_df:
            input_df[col] = 0

    input_df = input_df[feature_cols]

    # Prediction
    prob = model.predict_proba(input_df.values)[0, 1]
    risk = categorize_risk(prob)
    conf = confidence_level(prob)

    # Recommendations
    recs = generate_recommendations(
        age=data["age"],
        bmi=bmi,
        cycle_length_days=data["cycle_length_days"],
        cycle_irregular_flag=cycle_irregular_flag,
        symptom_score=symptom_score,
        hair_loss=data["hair_loss"],
        pimples=data["pimples"],
        skin_darkening=data["skin_darkening"],
        hair_growth=data["hair_growth"],
        exercise_level=data["exercise_level"],
        diet_quality=data["diet_quality"],
        stress_score=data["stress_score"],
        sleep_hours=data["sleep_hours"],
        risk_level=risk
    )

    return {
        "probability": float(prob),
        "risk_level": risk,
        "confidence": conf,
        "bmi": round(bmi, 1),
        "recommendations": recs
    }
