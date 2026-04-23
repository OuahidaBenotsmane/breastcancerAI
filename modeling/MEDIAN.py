import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.model_selection import GridSearchCV

# Load data
X_train = pd.read_csv("X_train_preprocessed.csv")
X_test = pd.read_csv("X_test_preprocessed.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Separate missing-indicator columns
indicator_cols = [col for col in X_train.columns if col.endswith("_missing")]
normal_cols = [col for col in X_train.columns if col not in indicator_cols]

print(f"Normal columns: {len(normal_cols)}")
print(f"Missing indicator columns: {len(indicator_cols)}")

# Median imputation on all columns
imputer = SimpleImputer(strategy="median")
X_train_med = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_med = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

print("NaN in X_train after median:", X_train_med.isnull().sum().sum())
print("NaN in X_test after median:", X_test_med.isnull().sum().sum())

# Scale only normal columns
scaler = StandardScaler()
X_train_scaled = X_train_med.copy()
X_test_scaled = X_test_med.copy()

X_train_scaled[normal_cols] = scaler.fit_transform(X_train_med[normal_cols])
X_test_scaled[normal_cols] = scaler.transform(X_test_med[normal_cols])

# Class imbalance ratio for XGBoost
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count

results = {}
all_preds = {}

# Logistic Regression
print("\n" + "=" * 50)
print("Tuning Logistic Regression...")
print("=" * 50)

lr_grid = GridSearchCV(
    LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
    {
        "C": [0.1, 1, 10],
        "solver": ["liblinear", "lbfgs"]
    },
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)
lr_grid.fit(X_train_scaled, y_train)

# SVM
print("\n" + "=" * 50)
print("Tuning SVM...")
print("=" * 50)

svm_grid = GridSearchCV(
    SVC(probability=True, class_weight="balanced", random_state=42),
    {
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"]
    },
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)
svm_grid.fit(X_train_scaled, y_train)

# Random Forest
print("\n" + "=" * 50)
print("Tuning Random Forest...")
print("=" * 50)

rf_grid = GridSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42),
    {
        "n_estimators": [200, 300],
        "max_depth": [5, 8, None],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 5]
    },
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)
rf_grid.fit(X_train_med, y_train)

# XGBoost
print("\n" + "=" * 50)
print("Tuning XGBoost...")
print("=" * 50)

xgb_grid = GridSearchCV(
    XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    ),
    {
        "n_estimators": [200, 300],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0]
    },
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)
xgb_grid.fit(X_train_med, y_train)

# LightGBM
print("\n" + "=" * 50)
print("Tuning LightGBM...")
print("=" * 50)

lgbm_grid = GridSearchCV(
    LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1),
    {
        "n_estimators": [200, 300],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 4, -1]
    },
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)
lgbm_grid.fit(X_train_med, y_train)

best_models = {
    "Logistic Regression": (lr_grid.best_estimator_, X_test_scaled),
    "SVM": (svm_grid.best_estimator_, X_test_scaled),
    "Random Forest": (rf_grid.best_estimator_, X_test_med),
    "XGBoost": (xgb_grid.best_estimator_, X_test_med),
    "LightGBM": (lgbm_grid.best_estimator_, X_test_med)
}

# Evaluate
for name, (model, X_te) in best_models.items():
    print("\n" + "=" * 50)
    print(f"{name} — Median Imputation")
    print("=" * 50)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    results[name] = {
        "Accuracy": round(acc, 4),
        "ROC-AUC": round(auc, 4),
        "F1": round(f1, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4)
    }

    all_preds[name] = {
        "y_pred": y_pred,
        "y_prob": y_prob
    }

    print(f"Best Params: {model.get_params()}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

# Save imputed data
X_train_med.to_csv("X_train_median_imputed.csv", index=False)
X_test_med.to_csv("X_test_median_imputed.csv", index=False)

# Summary
print("\n===== SUMMARY — Median Imputation =====")
summary = pd.DataFrame(results).T
print(summary.to_string())

summary.to_csv("results_median.csv")
print("\nSaved:")
print("- X_train_median_imputed.csv")
print("- X_test_median_imputed.csv")
print("- results_median.csv")

# Visualizations
print("\nGenerating plots...")

colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
model_names = list(results.keys())

# 1. Confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Confusion Matrices — Median Imputation", fontsize=14, fontweight="bold")

for ax, (name, preds), color in zip(axes.flat, all_preds.items(), colors):
    cm = confusion_matrix(y_test, preds["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Event", "Event"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    acc = results[name]["Accuracy"]
    ax.set_title(f"{name}\nAcc={acc:.4f}", fontsize=10)

for j in range(len(all_preds), len(axes.flat)):
    fig.delaxes(axes.flat[j])

plt.tight_layout()
plt.savefig("confusion_matrices_median.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: confusion_matrices_median.png")

# 2. ROC curves
fig, ax = plt.subplots(figsize=(8, 6))
for (name, preds), color in zip(all_preds.items(), colors):
    auc = results[name]["ROC-AUC"]
    RocCurveDisplay.from_predictions(
        y_test, preds["y_prob"],
        name=f"{name} (AUC={auc:.4f})",
        ax=ax, color=color
    )

ax.plot([0, 1], [0, 1], "k--", label="Random")
ax.set_title("ROC Curves — Median Imputation", fontweight="bold")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves_median.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: roc_curves_median.png")

# 3. Metrics comparison
metrics = ["Accuracy", "ROC-AUC", "F1", "Precision", "Recall"]
x = np.arange(len(model_names))
width = 0.15

fig, ax = plt.subplots(figsize=(14, 6))
for i, metric in enumerate(metrics):
    vals = [results[m][metric] for m in model_names]
    ax.bar(x + i * width, vals, width, label=metric)

ax.set_xticks(x + width * 2)
ax.set_xticklabels(model_names, rotation=15, ha="right")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Model Comparison — Median Imputation", fontweight="bold")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("metrics_comparison_median.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: metrics_comparison_median.png")

# Final comparison with other imputation methods
print("\n" + "=" * 60)
print(" FINAL COMPARISON — KNN vs MEDIAN vs MICE ")
print("=" * 60)

try:
    knn = pd.read_csv("results_knn_imputation.csv", index_col=0)
    median = pd.read_csv("results_median.csv", index_col=0)
    mice = pd.read_csv("results_mice.csv", index_col=0)

    knn["Method"] = "KNN"
    median["Method"] = "Median"
    mice["Method"] = "MICE"

    all_results = pd.concat([knn, median, mice])
    all_results = all_results.reset_index().rename(columns={"index": "Model"})

    print("\nBest model per method (based on ROC-AUC):")
    best_per_method = all_results.loc[
        all_results.groupby("Method")["ROC-AUC"].idxmax()
    ]
    print(best_per_method[["Method", "Model", "ROC-AUC", "F1", "Accuracy"]])

    best_overall = all_results.loc[all_results["ROC-AUC"].idxmax()]

    print("\n BEST OVERALL MODEL:")
    print(f"Method : {best_overall['Method']}")
    print(f"Model  : {best_overall['Model']}")
    print(f"ROC-AUC: {best_overall['ROC-AUC']}")
    print(f"F1     : {best_overall['F1']}")
    print(f"Acc    : {best_overall['Accuracy']}")

    all_results.to_csv("final_comparison_all_methods.csv", index=False)
    print("\nSaved: final_comparison_all_methods.csv")

except FileNotFoundError:
    print("\n Missing result files. Run all 3 scripts first (KNN, Median, MICE).")