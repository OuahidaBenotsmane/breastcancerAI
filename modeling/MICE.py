import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
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

# Scale only normal columns before MICE
scaler_before_mice = StandardScaler()
X_train_scaled_normal = scaler_before_mice.fit_transform(X_train[normal_cols])
X_test_scaled_normal = scaler_before_mice.transform(X_test[normal_cols])

# Recombine with indicator columns
if indicator_cols:
    X_train_before_mice = np.hstack([X_train_scaled_normal, X_train[indicator_cols].values])
    X_test_before_mice = np.hstack([X_test_scaled_normal, X_test[indicator_cols].values])
    all_cols = normal_cols + indicator_cols
else:
    X_train_before_mice = X_train_scaled_normal
    X_test_before_mice = X_test_scaled_normal
    all_cols = normal_cols

# MICE imputation
mice_imputer = IterativeImputer(
    max_iter=10,
    random_state=42,
    sample_posterior=True
)

X_train_mice = mice_imputer.fit_transform(X_train_before_mice)
X_test_mice = mice_imputer.transform(X_test_before_mice)

# Back to DataFrame
X_train_mice = pd.DataFrame(X_train_mice, columns=all_cols)
X_test_mice = pd.DataFrame(X_test_mice, columns=all_cols)

print("NaN in X_train after MICE:", X_train_mice.isnull().sum().sum())
print("NaN in X_test after MICE:", X_test_mice.isnull().sum().sum())

# Scale only normal columns again for models that need scaled inputs
scaler_for_models = StandardScaler()
X_train_scaled = X_train_mice.copy()
X_test_scaled = X_test_mice.copy()

X_train_scaled[normal_cols] = scaler_for_models.fit_transform(X_train_mice[normal_cols])
X_test_scaled[normal_cols] = scaler_for_models.transform(X_test_mice[normal_cols])

# Class imbalance ratio for XGBoost
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count

# Models
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    ),
    "SVM": SVC(
        probability=True,
        class_weight="balanced",
        C=1,
        kernel="rbf",
        gamma="scale",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        class_weight="balanced",
        random_state=42,
        verbose=-1
    )
}

# Models that need scaled data
needs_scaling = ["Logistic Regression", "SVM"]

# Train and evaluate
results = {}
all_preds = {}

for name, model in models.items():
    print("\n" + "=" * 50)
    print(f"{name} — MICE Imputation")
    print("=" * 50)

    if name in needs_scaling:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train_mice, X_test_mice

    model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)
    y_pred_prob = model.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
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
        "y_prob": y_pred_prob
    }

    print(f"Accuracy:  {acc:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

# Save imputed data
X_train_mice.to_csv("X_train_mice_imputed.csv", index=False)
X_test_mice.to_csv("X_test_mice_imputed.csv", index=False)

# Summary
print("\n===== SUMMARY — MICE Imputation =====")
summary = pd.DataFrame(results).T
print(summary.to_string())

summary.to_csv("results_mice.csv")
print("\nSaved:")
print("- X_train_mice_imputed.csv")
print("- X_test_mice_imputed.csv")
print("- results_mice.csv")

# Visualizations
print("\nGenerating plots...")

colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
model_names = list(results.keys())

# 1. Confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Confusion Matrices — MICE Imputation", fontsize=14, fontweight="bold")

for ax, (name, preds), color in zip(axes.flat, all_preds.items(), colors):
    cm = confusion_matrix(y_test, preds["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Event", "Event"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    acc = results[name]["Accuracy"]
    ax.set_title(f"{name}\nAcc={acc:.4f}", fontsize=10)

for j in range(len(all_preds), len(axes.flat)):
    fig.delaxes(axes.flat[j])

plt.tight_layout()
plt.savefig("confusion_matrices_mice.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: confusion_matrices_mice.png")

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
ax.set_title("ROC Curves — MICE Imputation", fontweight="bold")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves_mice.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: roc_curves_mice.png")

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
ax.set_title("Model Comparison — MICE Imputation", fontweight="bold")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("metrics_comparison_mice.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: metrics_comparison_mice.png")

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

    print("\n🏆 BEST OVERALL MODEL:")
    print(f"Method : {best_overall['Method']}")
    print(f"Model  : {best_overall['Model']}")
    print(f"ROC-AUC: {best_overall['ROC-AUC']}")
    print(f"F1     : {best_overall['F1']}")
    print(f"Acc    : {best_overall['Accuracy']}")

    all_results.to_csv("final_comparison_all_methods.csv", index=False)
    print("\nSaved: final_comparison_all_methods.csv")

except FileNotFoundError:
    print("\n⚠️ Missing result files. Run all 3 scripts first (KNN, Median, MICE).")