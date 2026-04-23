import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# Load data
df = pd.read_csv("metabric_patient_sample_merged.csv")

# Remove useless columns
cols_to_drop = [
    "PATIENT_ID", "SAMPLE_ID",
    "SEX", "CANCER_TYPE", "SAMPLE_TYPE",
    "OS_MONTHS", "OS_STATUS", "VITAL_STATUS", "RFS_MONTHS",
    "RFS_STATUS", "ER_STATUS", "ONCOTREE_CODE",
    "COHORT"
]
df = df.drop(columns=cols_to_drop, errors="ignore")

# Remove rows without target
df = df.dropna(subset=["recurred"])

# Fix text errors
df["ER_IHC"] = df["ER_IHC"].replace("Positve", "Positive")
df["CLAUDIN_SUBTYPE"] = df["CLAUDIN_SUBTYPE"].replace("claudin-low", "Claudin-low")
df["HER2_SNP6"] = df["HER2_SNP6"].replace("UNDEF", pd.NA)
df["CLAUDIN_SUBTYPE"] = df["CLAUDIN_SUBTYPE"].replace("NC", pd.NA)

#Her2 & Claudin subtype contradiction fix 
df = df[~((df["HER2_STATUS"] == "Positive") & (df["CLAUDIN_SUBTYPE"] == "LumA"))]
print(df[(df["HER2_STATUS"] == "Positive") & (df["CLAUDIN_SUBTYPE"] == "LumA")].shape[0])  # should be 0

#Claudin subtype & Er status contradiction fix
df = df[~((df["CLAUDIN_SUBTYPE"] == "Basal") & (df["ER_IHC"] == "Positive"))]
print(df[(df["CLAUDIN_SUBTYPE"] == "Basal") & (df["ER_IHC"] == "Positive")].shape[0])  # should be 0

# Drop Stage 1 but more than 3 positive lymph nodes
df = df[~((df["TUMOR_STAGE"] == 1) & (df["LYMPH_NODES_EXAMINED_POSITIVE"] > 3))]
print(f"Shape after dropping: {df.shape}")

#Drop Stage 4 but no positive lymph nodes
df = df[~((df["TUMOR_STAGE"] == 4) & (df["LYMPH_NODES_EXAMINED_POSITIVE"] == 0))]
print(df[(df["TUMOR_STAGE"] == 4) & (df["LYMPH_NODES_EXAMINED_POSITIVE"] == 0)].shape[0])  # should be 0

# Drop NPI (composite of GRADE + TUMOR_SIZE + LYMPH_NODES)
df = df.drop(columns=["NPI"])
print(f"Shape after dropping NPI: {df.shape}")


print("=== MISSING VALUES ===")
print(df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False))

# Drop only very high-missing columns
missing_pct = df.isnull().mean()
cols_to_drop = missing_pct[missing_pct > 0.70].index
df = df.drop(columns=cols_to_drop)

# Add missing indicators only for moderate missingness
missing_pct = df.isnull().mean()
for col in df.columns:
    if col != "recurred" and 0.05 < missing_pct[col] <= 0.70:
        df[col + "_missing"] = df[col].isnull().astype(int)

# Log transform
for col in ["TUMOR_SIZE", "LYMPH_NODES_EXAMINED_POSITIVE"]:
    if col in df.columns:
        df[col] = np.log1p(df[col])

# Clean target
df["recurred"] = pd.to_numeric(df["recurred"], errors="coerce")
df = df.dropna(subset=["recurred"])
df["recurred"] = df["recurred"].astype(int)

# Split
X = df.drop(columns=["recurred"])
y = df["recurred"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Ordinal encoding
ordinal_cols = {
    "CELLULARITY": ["Low", "Moderate", "High"],
    "GRADE": [1, 2, 3],
    "INFERRED_MENOPAUSAL_STATE": ["Pre", "Post"],
    "CLAUDIN_SUBTYPE": ["LumA", "LumB", "Normal", "Claudin-low", "Her2", "Basal"],
    "HER2_SNP6": ["LOSS", "NEUTRAL", "GAIN"],
}

ord_cols = [col for col in ordinal_cols if col in X_train.columns]

encoder = OrdinalEncoder(
    categories=[ordinal_cols[col] for col in ord_cols],
    handle_unknown="use_encoded_value",
    unknown_value=np.nan
)

if ord_cols:
    X_train[ord_cols] = encoder.fit_transform(X_train[ord_cols])
    X_test[ord_cols] = encoder.transform(X_test[ord_cols])

# One-hot encoding
nominal_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

X_train = pd.get_dummies(X_train, columns=nominal_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=nominal_cols, drop_first=True)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Save
X_train.to_csv("X_train_preprocessed.csv", index=False)
X_test.to_csv("X_test_preprocessed.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Preprocessing done!")