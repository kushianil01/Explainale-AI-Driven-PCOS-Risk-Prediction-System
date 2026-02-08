"""
PCOS Balanced Synthetic Model - v3
----------------------------------
- Loads your synthetic PCOS dataset
- Trains an optimized XGBoost classifier
- Evaluates using AUC, Accuracy, Recall, F1
- Applies SHAP summary & feature importance plot
- Zero SHAP errors due to safe wrapper
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import shap
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

# 🔽 Load dataset
print("📂 Loading synthetic dataset...")
df = pd.read_csv("synthetic_pcos_balanced_v2.csv")
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("PCOS distribution:", df['pcos'].value_counts(normalize=True).round(3).to_dict())

# Define features and labels
X = df.drop("pcos", axis=1)
y = df['pcos']

# 🔀 Train-test split (no SMOTE needed because data is balanced)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain size:", len(X_train))
print("Test size:", len(X_test))

# ⚙ Train XGBoost Model
print("\n⚙ Training XGBoost model on synthetic data...")
model = XGBClassifier(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.85,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 🧪 Model Evaluation
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

print("\n===== Test Set Performance =====")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))

# 🔁 Cross-validation
print("\n===== 5-Fold Cross Validation (AUC) =====")
cv_auc = cross_val_score(
    model, X, y,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='roc_auc'
)
print("AUC scores:", np.round(cv_auc, 3))
print(f"Mean AUC: {cv_auc.mean():.3f}")

# -----------------------------  
# -----------------------------
# 📊 Feature Importance (ELI5)
# -----------------------------
import eli5
from eli5.sklearn import PermutationImportance

print("\n🧠 Generating Explainable AI Report via ELI5...")

# Use permutation-based feature importance (works with XGBoost)
perm = PermutationImportance(model, random_state=42).fit(X_test, y_test)

# Save importance to HTML
with open("pcos_feature_importance.html", "w") as f:
    f.write(eli5.format_as_html(eli5.explain_weights(perm, feature_names=X.columns.tolist())))
    print("✔ Feature importance saved as 'pcos_feature_importance.html'")
