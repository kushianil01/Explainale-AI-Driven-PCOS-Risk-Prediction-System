from fastapi import FastAPI
import pandas as pd
from pcos_app import train_pcos_model, categorize_risk, confidence_level

app = FastAPI()

model, feature_cols, _, _ = train_pcos_model()

@app.get("/")
def home():
    return {"message": "PCOS API is running"}

@app.post("/predict")
def predict(data: dict):
    input_df = pd.DataFrame([data])

    for col in feature_cols:
        if col not in input_df:
            input_df[col] = 0

    input_df = input_df[feature_cols]

    prob = model.predict_proba(input_df.values)[0, 1]

    return {
        "probability": float(prob),
        "risk_level": categorize_risk(prob),
        "confidence": confidence_level(prob)
    }