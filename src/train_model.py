import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import joblib


# -------------------------
# Paths & constants
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "post-operative-data.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "post_op_lr_pipeline.pkl"
METRICS_PATH = MODEL_DIR / "post_op_lr_metrics.txt"

TARGET_COL_RAW = "decision ADM-DECS"
THRESHOLD = 0.3  # tuned threshold for class 1 (complication)


# -------------------------
# Data loading & cleaning
# -------------------------
def load_and_clean_data(csv_path: Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    print(f"📥 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    print("Original shape:", df.shape)

    # Strip column names
    df.columns = df.columns.str.strip()

    # Clean target column values (remove spaces)
    if TARGET_COL_RAW not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL_RAW}' not found in columns: {df.columns.tolist()}")

    df[TARGET_COL_RAW] = df[TARGET_COL_RAW].astype(str).str.strip()

    # Replace '?' with NaN everywhere
    df.replace("?", np.nan, inplace=True)

    # Convert COMFORT to numeric if present
    if "COMFORT" in df.columns:
        df["COMFORT"] = pd.to_numeric(df["COMFORT"], errors="coerce")

    # Drop rows where target is missing
    df = df[df[TARGET_COL_RAW].notna()]

    # Keep only expected labels
    df = df[df[TARGET_COL_RAW].isin(["A", "S", "I"])]

    # Binary mapping: A = 0 (safe), S/I = 1 (needs monitoring)
    df[TARGET_COL_RAW] = df[TARGET_COL_RAW].replace({"A": 0, "S": 1, "I": 1})

    # Drop rows with missing COMFORT (if any)
    if "COMFORT" in df.columns:
        df = df.dropna(subset=["COMFORT"])

    print("Cleaned shape:", df.shape)
    print("Target distribution after mapping:")
    print(df[TARGET_COL_RAW].value_counts())

    X = df.drop(columns=[TARGET_COL_RAW])
    y = df[TARGET_COL_RAW]

    return X, y


# -------------------------
# Model training
# -------------------------
def train_model():
    # 1. Load & clean data
    X, y = load_and_clean_data()

    # 2. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTrain size:", X_train.shape, " Test size:", X_test.shape)

    # 3. Identify categorical columns (objects) and numeric columns
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(exclude=["object"]).columns.tolist()

    print("\nCategorical columns:", categorical_cols)
    print("Numeric columns:", numeric_cols)

    # 4. Preprocessor: OneHot encode categoricals, pass-through numeric
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ],
        remainder="passthrough",  # numeric columns go through unchanged
    )

    # 5. Logistic Regression model (baseline with class_weight)
    log_reg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    # 6. Full pipeline: preprocessing + model
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", log_reg),
        ]
    )

    # 7. Fit model
    print("\n🚀 Training Logistic Regression model...")
    clf.fit(X_train, y_train)

    # 8. Evaluate
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred_default = clf.predict(X_test)  # threshold = 0.5
    y_pred_tuned = (y_proba >= THRESHOLD).astype(int)  # threshold = 0.3

    print("\n=== Evaluation with default threshold (0.5) ===")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_default))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred_default))

    print("\n=== Evaluation with tuned threshold (0.3) ===")
    cm_tuned = confusion_matrix(y_test, y_pred_tuned)
    print("Confusion matrix:")
    print(cm_tuned)
    print("\nClassification report:")
    cr_tuned = classification_report(y_test, y_pred_tuned)
    print(cr_tuned)

    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC (probability-based): {auc:.3f}")

    # 9. Save metrics to a text file
    with open(METRICS_PATH, "w") as f:
        f.write("=== Logistic Regression with tuned threshold ===\n\n")
        f.write("Confusion matrix (threshold = 0.3):\n")
        f.write(str(cm_tuned) + "\n\n")
        f.write("Classification report (threshold = 0.3):\n")
        f.write(cr_tuned + "\n")
        f.write(f"ROC-AUC: {auc:.3f}\n")

    print(f"\n📄 Metrics saved to: {METRICS_PATH}")

    # 10. Save model + threshold as one object
    artifact = {
        "pipeline": clf,
        "threshold": THRESHOLD,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"✅ Trained model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
