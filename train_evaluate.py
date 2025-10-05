import argparse
import io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from joblib import dump
import matplotlib.pyplot as plt

NUMERIC_COLS_DEFAULT = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_teq", "koi_insol", "koi_model_snr", "koi_impact",
    "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag"
]

def load_koi_csv(path: str) -> pd.DataFrame:
    # NASA KOI CSVs start with comment lines beginning with '#'
    return pd.read_csv(path, comment="#", low_memory=False)

def train_model(df: pd.DataFrame, target_col: str, numeric_cols=None, test_size=0.2, random_state=42):
    if numeric_cols is None:
        numeric_cols = NUMERIC_COLS_DEFAULT

    # Normalize label to canonical classes
    y = df[target_col].astype(str).str.upper().str.strip()
    # Keep just relevant classes if present
    keep = y.isin(["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"])
    df = df.loc[keep].copy()
    y = y.loc[keep]

    X = df[numeric_cols].copy()
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Train RF
    rf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)

    # Feature importances
    importances = rf.feature_importances_
    feat_imp = sorted(zip(numeric_cols, importances), key=lambda x: x[1], reverse=True)

    print("Rows used:", len(df))
    print("Classes:", list(rf.classes_))
    print("Accuracy:", round(acc, 3))
    print("Weighted F1:", round(f1, 3))
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print("\nTop features:")
    for name, val in feat_imp[:10]:
        print(f"  {name}: {val:.3f}")

    # Save artifacts
    dump({"model": rf, "imputer": imp, "features": numeric_cols, "classes": rf.classes_}, "kepler_model.joblib")
    print("\nSaved model => kepler_model.joblib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to Kepler KOI cumulative CSV")
    parser.add_argument("--target", default="koi_pdisposition", help="Label column to use")
    args = parser.parse_args()

    df = load_koi_csv(args.csv)
    train_model(df, target_col=args.target)
