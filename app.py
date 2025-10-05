import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="ExoFindr — Kepler KOI Demo", layout="wide")
st.title("ExoFindr — Kepler Exoplanet Classifier")
st.caption("NASA Space Apps 2025 — Kepler KOI cumulative (Disposition Using Kepler Data)")

st.markdown("""Upload the **Kepler KOI cumulative CSV**. We'll skip the NASA comment lines, select numeric features,
train a RandomForest to classify **CANDIDATE vs FALSE POSITIVE (and CONFIRMED if present)**,
and show metrics, a confusion matrix, and feature importance.
""")

NUMERIC_COLS = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_teq", "koi_insol", "koi_model_snr", "koi_impact",
    "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag"
]

with st.sidebar:
    st.header("Settings")
    target_col = st.selectbox("Label column", ["koi_pdisposition", "koi_disposition"])
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    n_estimators = st.slider("n_estimators", 50, 500, 200, 50)
    max_depth = st.slider("max_depth (0=None)", 0, 50, 0, 1)
    min_samples_split = st.slider("min_samples_split", 2, 20, 2, 1)

uploaded = st.file_uploader("Upload Kepler KOI cumulative CSV", type=["csv"])

def load_csv(file):
    return pd.read_csv(file, comment="#", low_memory=False)

def normalize_label(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper().str.strip()
    s = s.replace({
        "CONFIRMED": "CONFIRMED",
        "CANDIDATE": "CANDIDATE",
        "FALSE POSITIVE": "FALSE POSITIVE",
        "FP": "FALSE POSITIVE",
        "PC": "CANDIDATE"
    })
    return s

if uploaded:
    df = load_csv(uploaded)
    st.write("**Preview**", df.head())

    if target_col not in df.columns:
        st.error(f"Target column `{target_col}` not found in CSV.")
        st.stop()

    y = normalize_label(df[target_col])
    keep = y.isin(["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"])
    df = df.loc[keep].copy()
    y = y.loc[keep]

    st.write("**Class balance**", y.value_counts())

    X = df[NUMERIC_COLS].copy()
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=test_size, stratify=y, random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None if max_depth==0 else max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{acc:.3f}")
    col2.metric("Weighted F1", f"{f1:.3f}")

    st.text("Classification report")
    st.code(classification_report(y_test, y_pred), language="text")

    # Confusion matrix
    labels = list(rf.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels, rotation=20)
    plt.yticks(range(len(labels)), labels)
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, str(z), ha='center', va='center')
    st.pyplot(fig)

    # Feature importance
    importances = rf.feature_importances_
    inds = np.argsort(importances)[::-1][:15]
    fig2 = plt.figure()
    plt.barh(range(len(inds))[::-1], importances[inds][::-1])
    names = [np.array(NUMERIC_COLS)[inds][i] for i in range(len(inds))][::-1]
    plt.yticks(range(len(inds))[::-1], names)
    plt.title("Top Feature Importances")
    st.pyplot(fig2)

    # Predict on new data
    st.subheader("Predict on new rows (optional)")
    new = st.file_uploader("Upload unlabeled CSV with same numeric columns", type=["csv"], key="pred")
    if new:
        new_df = pd.read_csv(new, comment="#", low_memory=False)
        new_X = new_df.reindex(columns=NUMERIC_COLS)
        new_imp = imp.transform(new_X)
        preds = rf.predict(new_imp)
        out = new_df.copy()
        out["prediction"] = preds
        st.write(out.head())
        buf = io.StringIO()
        out.to_csv(buf, index=False)
        st.download_button("Download predictions.csv", buf.getvalue(), "predictions.csv", "text/csv")
else:
    st.info("Upload the KOI cumulative CSV to begin.")
