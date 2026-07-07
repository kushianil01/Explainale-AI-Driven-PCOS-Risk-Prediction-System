# Explainale-AI-Driven-PCOS-Risk-Prediction-System

🩺 Explainable AI-Driven PCOS Risk Prediction System
Overview
The Explainable AI-Driven PCOS Risk Prediction System is a machine learning web application that predicts the likelihood of Polycystic Ovary Syndrome (PCOS) using non-invasive clinical parameters. Unlike conventional prediction systems, this project provides interpretable explanations for every prediction using Explainable AI, allowing users to understand which factors contributed most to the predicted risk.

Features
PCOS Risk Prediction
Explainable AI using ELI5
Confidence Score
Real-time Prediction
User-friendly Streamlit Interface
Non-invasive Diagnosis
Interactive Dashboard
Tech Stack
Python
XGBoost
Scikit-learn
ELI5
Streamlit
Pandas
NumPy

Project Workflow
Collect patient information.
Preprocess the data.
Train an XGBoost classifier.
Generate PCOS prediction.
Explain predictions using ELI5.
Display prediction confidence and feature importance.

Project Structure
PCOS-Risk-Prediction/
│
├── data/
│
├── models/
│   ├── xgboost_model.pkl
│
├── app.py
├── train.py
├── preprocess.py
├── requirements.txt
└── README.md

Model Performance
ROC-AUC Score: 0.90
Recall: 85%

Input Features
Age
BMI
Weight
Height
Menstrual Cycle Regularity
Acne
Hair Growth
Weight Gain
Skin Darkening
Physical Activity

Installation
git clone https://github.com/yourusername/PCOS-Risk-Prediction.git

cd PCOS-Risk-Prediction

pip install -r requirements.txt

Run
streamlit run app.py

Output
The application provides:

PCOS Risk Prediction
Prediction Confidence
Feature Importance
Explainable AI Visualization
Future Improvements
SHAP-based Explainability
Deep Learning Models
Cloud Deployment
Electronic Health Record Integration
Multi-Class Risk Assessment

Author
Kushi Anil Kumbar
