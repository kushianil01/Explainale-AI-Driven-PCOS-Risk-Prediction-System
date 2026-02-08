"""
PCOS Risk Prediction Model + Explainable AI using ELI5
v3 Final — Fully Working + HTML Explanations Included

Outputs:
    /results/
        roc_curve.png
        confusion_matrix.png
        feature_importance.png
        eli5_global_feature_importance.html
        eli5_local_example.html
        metrics.txt
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay
)
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from eli5.sklearn import PermutationImportance
import eli5
import seaborn as sns
warnings = __import__("warnings")
warnings.filterwarnings("ignore")

# ===============================
# CONFIG
# ===============================
DATA_PATH = "synthetic_pcos_balanced_v2.csv"   # <-- change to your dataset
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===============================
# LOAD DATA
# ===============================
print("\n📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("PCOS distribution:", df["pcos"].value_counts(normalize=True).round(3).to_dict())

# ===============================
# SPLIT DATA
# ===============================
feature_cols = [c for c in df.columns if c != "pcos"]
X = df[feature_cols]
y = df["pcos"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("\nTrain size:", len(X_train))
print("Test size:", len(X_test))

# ===============================
# TRAIN MODEL
# ===============================
print("\n⚙ Training XGBoost model...")
model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

# ===============================
# METRICS
# ===============================
print("\n===== Test Set Performance =====")
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1-Score: {f1:.3f}")
print(f"ROC-AUC: {auc:.3f}\n")

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))

# ===== Save basic metrics =====
with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
    f.write(f"Accuracy: {acc:.3f}\nPrecision: {prec:.3f}\nRecall: {rec:.3f}\n"
            f"F1: {f1:.3f}\nROC-AUC: {auc:.3f}\n\n")
    f.write(classification_report(y_test, y_pred))

# ===============================
# Cross-Validation
# ===============================
print("\n===== 5-Fold Cross Validation (AUC) =====")
cv_auc = cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc")
print("AUC scores:", np.round(cv_auc, 3))
print("Mean AUC:", round(cv_auc.mean(), 3))

# ===============================
# ROC Curve Plot
# ===============================
plt.figure(figsize=(6, 5))
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("ROC Curve - PCOS Risk Model")
plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=300)
plt.close()

# ===============================
# Confusion Matrix Plot
# ===============================
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

# ===============================
# Feature Importance
# ===============================
plt.figure(figsize=(6, 6))
imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
imp[:15].plot(kind="barh")
plt.title("Top Feature Importances (XGBoost)")
plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=300)
plt.close()

# ===============================
# EXPLAINABLE AI (ELI5)
# ===============================
print("\n🧠 Generating ELI5 explainability outputs...")

# ---- GLOBAL EXPLANATION ----
perm = PermutationImportance(model, random_state=42, n_iter=10)
perm.fit(X_test, y_test)

global_exp = eli5.explain_weights(perm, feature_names=feature_cols)
global_html = eli5.format_as_html(global_exp)
with open(os.path.join(RESULTS_DIR, "eli5_global_feature_importance.html"), "w", encoding="utf-8") as f:
    f.write(global_html)

# ---- LOCAL EXPLANATION ----
local_idx = 3
local_exp = eli5.explain_prediction(model, X_test.iloc[local_idx], feature_names=feature_cols)
local_html = eli5.format_as_html(local_exp)
with open(os.path.join(RESULTS_DIR, "eli5_local_example.html"), "w", encoding="utf-8") as f:
    f.write(local_html)

print("\n🎉 DONE — All results saved in /results/")
