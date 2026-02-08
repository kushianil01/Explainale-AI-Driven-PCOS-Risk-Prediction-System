import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ==========================
# 1. Load Dataset
# ==========================
df = pd.read_csv("synthetic_pcos_balanced_v2.csv")
print("Loaded:", df.shape)

X = df.drop("pcos", axis=1)
y = df["pcos"]
feature_cols = list(X.columns)

# ==========================
# 2. Train Model
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ==========================
# 3. Evaluate Performance
# ==========================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n===== Performance =====")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("F1     :", round(f1_score(y_test, y_pred), 3))
print("AUC    :", round(roc_auc_score(y_test, y_proba), 3))

# ==========================
# 4. Feature Importance Plot
# ==========================
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 15
top_features = [feature_cols[i] for i in indices[:top_n]]
top_importance = importances[indices[:top_n]]

plt.figure(figsize=(10, 6))
sns.barplot(x=top_importance, y=top_features, palette="Blues_r")
plt.title("Top Feature Importances (XGBoost)", fontsize=16, fontweight="bold")
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.savefig("feature_importance_clean.png", dpi=300)
plt.show()

print("\n✅ Feature importance saved as feature_importance_clean.png")

# ==========================
# 5. Save Model for Later Use
# ==========================
joblib.dump(model, "pcos_xgb_trained.pkl")
print("💾 Model saved as pcos_xgb_trained.pkl")
