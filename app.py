import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="ExoFindr — Exoplanet Classifier", layout="wide")
st.title("ExoFindr — Exoplanet Classifier (Trained on Kepler)")
st.caption("NASA Space Apps 2025 — Model is pre-trained from repository data. Users enter values manually to get predictions.")

# -------------------------
# Configuration
# -------------------------
DATA_PATHS_TRY = [
    "data/kepler_cumulative.csv",       # put your KOI cumulative here
    "data/cumulative.csv",              # fallback name
    "data/KOI_cumulative.csv"           # another common name
]

NUMERIC_COLS = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_teq", "koi_insol", "koi_model_snr", "koi_impact",
    "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag"
]

POSSIBLE_TARGETS = ["koi_pdisposition", "koi_disposition", "disposition", "tfopwg_disp", "k2_disposition"]

def normalize_label(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper().str.strip()
    return s.replace({
        "CONFIRMED": "CONFIRMED",
        "CANDIDATE": "CANDIDATE",
        "PC": "CANDIDATE",
        "FALSE POSITIVE": "FALSE POSITIVE",
        "FP": "FALSE POSITIVE"
    })

def load_repo_csv() -> pd.DataFrame:
    for p in DATA_PATHS_TRY:
        if os.path.exists(p):
            # KOI/TOI often have '#' commented headers
            try:
                return pd.read_csv(p, comment="#", low_memory=False)
            except Exception:
                return pd.read_csv(p, low_memory=False)
    raise FileNotFoundError(
        f"No training CSV found. Add one of these to the repo: {', '.join(DATA_PATHS_TRY)}"
    )

def can_stratify(y: pd.Series) -> bool:
    vc = y.value_counts()
    return (vc.size >= 2) and (vc.min() >= 2)

@st.cache_resource(show_spinner=True)
def train_from_repo(
    test_size: float = 0.2,
    n_estimators: int = 200,
    max_depth: int = 0,
    min_samples_split: int = 2,
    random_state: int = 42,
):
    df = load_repo_csv()

    # auto-detect target
    tcol = next((c for c in POSSIBLE_TARGETS if c in df.columns), None)
    if not tcol:
        raise ValueError(f"Could not find label column. Add one of: {POSSIBLE_TARGETS}")

    # choose features that actually exist
    used_feats = [c for c in NUMERIC_COLS if c in df.columns]
    if len(used_feats) < 3:
        raise ValueError("Not enough numeric columns present. Expected some of: " + ", ".join(NUMERIC_COLS))

    y = normalize_label(df[tcol])
    keep = y.isin(["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"])
    if keep.sum() == 0:
        keep = y.notna()
    df = df.loc[keep].copy()
    y = y.loc[keep]

    if y.nunique() < 2:
        raise ValueError("Training data has only one class. Please include at least two classes (e.g., CANDIDATE and FALSE POSITIVE).")

    X = df.reindex(columns=used_feats)
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    stratify_opt = y if can_stratify(y) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=test_size, stratify=stratify_opt, random_state=random_state
    )

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None if max_depth == 0 else max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    return {
        "rf": rf,
        "imp": imp,
        "classes_": list(rf.classes_),
        "features": used_feats,
        "target_col": tcol,
        "metrics": {"accuracy": acc, "weighted_f1": f1},
        "y_test": y_test,
        "y_pred": y_pred,
        "train_rows": X_train.shape[0]
    }

# -------------------------
# Train (from repo data) + Show metrics
# -------------------------
try:
    model = train_from_repo()
except Exception as e:
    st.error(f"Setup error: {e}")
    st.stop()

m = model["metrics"]
c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{m['accuracy']:.3f}")
c2.metric("Weighted F1", f"{m['weighted_f1']:.3f}")
c3.metric("Training rows", f"{model['train_rows']}")

st.text("Classification report")
st.code(classification_report(model["y_test"], model["y_pred"]), language="text")

# Confusion matrix
st.subheader("Confusion Matrix")
labels = model["classes_"]
cm = confusion_matrix(model["y_test"], model["y_pred"], labels=labels)
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
st.subheader("Top Feature Importances")
importances = model["rf"].feature_importances_
inds = np.argsort(importances)[::-1][:15]
feat_names = np.array(model["features"])[inds]
fig2 = plt.figure()
plt.barh(range(len(inds))[::-1], importances[inds][::-1])
plt.yticks(range(len(inds))[::-1], feat_names[::-1])
plt.title("Top Features")
st.pyplot(fig2)

# -------------------------
# Manual entry prediction only
# -------------------------
st.subheader("Manual Prediction — Enter feature values")
st.caption("Tip: Use realistic values from Kepler KOI ranges; defaults are medians from the training data.")

# defaults from training medians
med_defaults = dict(zip(model["features"], model["imp"].statistics_))

with st.form("manual_form"):
    cols = st.columns(3)
    inputs = {}
    for i, feat in enumerate(model["features"]):
        val = float(med_defaults.get(feat, 0.0))
        # Use number_input; you can set min/max if you want stricter ranges
        inputs[feat] = cols[i % 3].number_input(feat, value=val)
    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame([inputs])[model["features"]]
    row_imp = model["imp"].transform(row)
    pred = model["rf"].predict(row_imp)[0]
    st.success(f"Predicted class: **{pred}**")

    if hasattr(model["rf"], "predict_proba"):
        proba = model["rf"].predict_proba(row_imp)[0]
        prob_map = dict(zip(model["rf"].classes_, proba))
        st.write("Class probabilities:")
        for cls, p in prob_map.items():
            st.write(f"- {cls}: {p:.3f}")

st.caption("Model trained from repository CSV. No user file uploads required.")
