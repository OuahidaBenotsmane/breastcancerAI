import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, classification_report
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

# Scale only normal columns before KNN
scaler_before_knn = StandardScaler()

X_train_scaled_normal = scaler_before_knn.fit_transform(X_train[normal_cols])
X_test_scaled_normal = scaler_before_knn.transform(X_test[normal_cols])

# Recombine with indicator columns
if indicator_cols:
    X_train_before_knn = np.hstack([X_train_scaled_normal, X_train[indicator_cols].values])
    X_test_before_knn = np.hstack([X_test_scaled_normal, X_test[indicator_cols].values])
    all_cols = normal_cols + indicator_cols
else:
    X_train_before_knn = X_train_scaled_normal
    X_test_before_knn = X_test_scaled_normal
    all_cols = normal_cols

# KNN imputation
knn_imputer = KNNImputer(n_neighbors=5)
X_train_knn = knn_imputer.fit_transform(X_train_before_knn)
X_test_knn = knn_imputer.transform(X_test_before_knn)

# Back to DataFrame
X_train_knn = pd.DataFrame(X_train_knn, columns=all_cols)
X_test_knn = pd.DataFrame(X_test_knn, columns=all_cols)

print("NaN in X_train after KNN:", X_train_knn.isnull().sum().sum())
print("NaN in X_test after KNN:", X_test_knn.isnull().sum().sum())

# Scale only normal columns again for models that need scaled inputs
scaler_for_models = StandardScaler()
X_train_knn[normal_cols] = scaler_for_models.fit_transform(X_train_knn[normal_cols])
X_test_knn[normal_cols] = scaler_for_models.transform(X_test_knn[normal_cols])

# Final scaled version for LR and SVM
X_train_final_scaled = X_train_knn.copy()
X_test_final_scaled = X_test_knn.copy()

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

for name, model in models.items():
    print("\n" + "=" * 50)
    print(f"{name} — KNN Imputation")
    print("=" * 50)

    if name in needs_scaling:
        X_tr, X_te = X_train_final_scaled, X_test_final_scaled
    else:
        X_tr, X_te = X_train_knn, X_test_knn

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

    print(f"Accuracy:  {acc:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

# Save imputed data
X_train_knn.to_csv("X_train_knn_imputed.csv", index=False)
X_test_knn.to_csv("X_test_knn_imputed.csv", index=False)

# Summary
print("\n===== SUMMARY — KNN Imputation =====")
summary = pd.DataFrame(results).T
print(summary.to_string())

summary.to_csv("results_knn_imputation.csv")
print("\nSaved:")
print("- X_train_knn_imputed.csv")
print("- X_test_knn_imputed.csv")
print("- results_knn_imputation.csv")

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