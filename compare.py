import matplotlib.pyplot as plt
import pandas as pd

# Define table content
data = {
    "Study": [
        "Elmannai et al. (2023)",
        "Zad et al. (2024)",
        "Zigarelli et al. (2022)",
        "This Work (2025)"
    ],
    "Dataset Source & Size": [
        "Benchmark dataset (~hundreds)",
        "Large EHR (30,601 patients)",
        "Survey dataset (~hundreds)",
        "Synthetic Balanced (2,000 samples)"
    ],
    "Features Used": [
        "Mixed clinical + symptoms",
        "Demographics + imaging + hormones",
        "Non-invasive questionnaire",
        "Non-invasive + engineered interactions"
    ],
    "Models Used": [
        "Stacked Ensemble (RF, XGB, SVM)",
        "LR, SVM, GBM, Neural Networks",
        "RF, SVM, LR",
        "XGBoost"
    ],
    "Explainability": [
        "ELI5, feature selection",
        "Feature ranking, clinical analysis",
        "Simple feature importance",
        "ELI5 global + local"
    ],
    "Key Metrics": [
        "Very high accuracy on small dataset",
        "AUC ≈ 0.80–0.85",
        "Accuracy ≈ 81–82%",
        "AUC ≈ 0.90 (synthetic)"
    ]
}

df = pd.DataFrame(data)

# Plot figure
fig, ax = plt.subplots(figsize=(16, 4))
ax.axis('off')

table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.3, 2.0)

plt.savefig("comparison_table.png", dpi=300, bbox_inches='tight')
plt.close()

print("Saved as comparison_table.png")
