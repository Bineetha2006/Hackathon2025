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
st.title("ExoFindr — Exoplanet Classifier (Kepler / K2 / TESS)")
st.caption("NASA Space Apps 2025 — Upload data, train, predict in batch, or manually enter values.")

# -------------------------
# Configuration
# -------------------------
NUMERIC_COLS = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad",
    "koi_teq", "koi_insol", "koi_model_snr", "koi_impact",
    "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag"
]

POSSIBLE_TARGETS = ["koi_pdisposition", "koi_disposition", "disposition", "tfopwg_disp", "k2_disposition"]

def normalize_label(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper().str.strip()
    # normalize common variants
    return s.replace({
        "CONFIRMED": "CONFIRMED",
        "CANDIDATE": "CANDIDATE",
        "PC": "CANDIDATE",
        "FALSE POSITIVE": "FALSE POSITIVE",
        "FP": "FALSE POSITIVE"
    })

def load_any_csv(file):
    # KOI/TOI often start with comment lines '#'
    try:
        df = pd.read_csv(file, comment="#", low_memory=False)
    except Exception:
        file.seek(0)
        df = pd.read_csv(file, low_memory=False)
    return df

@st.cache_resource(show_spinner=False)
def train_pipeline(df: pd.DataFrame, target_col: str, test_size: float, n_estimators:int,
                   max_depth:int, min_samples_split:int, random_state:int=42):
    # normalize labels & filter to our 3 canonical classes if present
    y = normalize_label(df[target_col])
    keep = y.isin(["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"])
    if keep.sum() == 0:
        # fallback: keep all labels present
        keep = y.notna()
    df = df.loc[keep].copy()
    y = y.loc[keep]

    # numeric features only
    X = df.reindex(columns=NUMERIC_COLS)
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # model
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

    # collect artifacts
    results = {
        "rf": rf,
        "imp": imp,
        "classes_": list(rf.classes_),
        "X_columns": NUMERIC_COLS,
        "X_train_shape": X_train.shape,
        "metrics": {"accuracy": acc, "weighted_f1": f1},
        "y_test": y_test,
        "y_pred": y_pred
    }
    return results

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("Settings")
    target_choice = st.selectbox(
        "Label column (choose what exists in your CSV)",
        ["auto-detect"] + POSSIBLE_TARGETS
    )
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    n_estimators = st.slider("RandomForest n_estimators", 50, 500, 200, 50)
    max_depth = st.slider("max_depth (0=None)", 0, 50, 0, 1)
    min_samples_split = st.slider("min_samples_split", 2, 20, 2, 1)
    show_probs = st.checkbox("Show class probabilities", True)

# -------------------------
# 1) Upload labeled data (training)
# -------------------------
st.subheader("1) Upload **labeled** CSV to train")
uploaded = st.file_uploader("Upload Kepler/K2/TESS catalog CSV (KOI/TOI).", type=["csv"])

model = None
if uploaded:
    df = load_any_csv(uploaded)
    st.write("**Preview (first 5 rows)**")
    st.dataframe(df.head())

    # pick target
    if target_choice == "auto-detect":
        tcol = next((c for c in POSSIBLE_TARGETS if c in df.columns), None)
    else:
        tcol = target_choice if target_choice in df.columns else None

    if not tcol:
        st.error(f"Could not find a label column. Try one of: {POSSIBLE_TARGETS}")
        st.stop()

    # Only keep numeric feature cols that exist in this CSV
    existing_feats = [c for c in NUMERIC_COLS if c in df.columns]
    if len(existing_feats) < 3:
        st.error("Not enough numeric columns found. Expected some of: " + ", ".join(NUMERIC_COLS))
        st.stop()

    # Train
    with st.spinner("Training model..."):
        res = train_pipeline(
            df[df.columns.intersection([tcol] + existing_feats)],
            target_col=tcol,
            test_size=test_size,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )
    model = res

    # Metrics
    m = res["metrics"]
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{m['accuracy']:.3f}")
    c2.metric("Weighted F1", f"{m['weighted_f1']:.3f}")

    st.text("Classification report")
    st.code(classification_report(res["y_test"], res["y_pred"]), language="text")

    # Confusion matrix
    st.subheader("Confusion Matrix")
    labels = res["classes_"]
    cm = confusion_matrix(res["y_test"], res["y_pred"], labels=labels)
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
    importances = res["rf"].feature_importances_
    # keep only existing features
    feat_names = [c for c in NUMERIC_COLS if c in df.columns]
    inds = np.argsort(importances)[::-1][:15]
    fig2 = plt.figure()
    plt.barh(range(len(inds))[::-1], importances[inds][::-1])
    names = [np.array(feat_names)[inds][i] for i in range(len(inds))][::-1]
    plt.yticks(range(len(inds))[::-1], names)
    plt.title("Top Features")
    st.pyplot(fig2)

# -------------------------
# 2) Manual entry form (single prediction)
# -------------------------
st.subheader("2) Manually enter values to predict a single case")
if model is None:
    st.info("Upload a labeled CSV and train a model first to enable manual predictions.")
else:
    # defaults from training medians where possible
    med_defaults = {}
    try:
        # Build a small DataFrame from the imputer statistics
        med_defaults = dict(zip(model["X_columns"], model["imp"].statistics_))
    except Exception:
        med_defaults = {c: 0.0 for c in model["X_columns"]}

    with st.form("manual_form"):
        cols = st.columns(3)
        inputs = {}
        for i, feat in enumerate(model["X_columns"]):
            val = med_defaults.get(feat, 0.0)
            # sensible ranges; users can edit freely
            inputs[feat] = cols[i % 3].number_input(feat, value=float(val))
        submitted = st.form_submit_button("Predict")

    if submitted:
        row = pd.DataFrame([inputs])[model["X_columns"]]
        row_imp = model["imp"].transform(row)
        pred = model["rf"].predict(row_imp)[0]
        st.success(f"Predicted class: **{pred}**")
        if show_probs and hasattr(model["rf"], "predict_proba"):
            proba = model["rf"].predict_proba(row_imp)[0]
            prob_map = dict(zip(model["rf"].classes_, proba))
            st.write("Class probabilities:")
            for cls, p in prob_map.items():
                st.write(f"- {cls}: {p:.3f}")

# -------------------------
# 3) Batch prediction (unlabeled CSV)
# -------------------------
st.subheader("3) Batch predict on a new (unlabeled) CSV")
if model is None:
    st.info("Train a model first to enable batch predictions.")
else:
    new_file = st.file_uploader("Upload unlabeled CSV with similar numeric columns", type=["csv"], key="pred")
    if new_file is not None:
        new_df = load_any_csv(new_file)
        st.write("**Preview**")
        st.dataframe(new_df.head())

        # align columns
        new_X = new_df.reindex(columns=model["X_columns"])
        new_imp = model["imp"].transform(new_X)
        preds = model["rf"].predict(new_imp)

        out = new_df.copy()
        out["prediction"] = preds
        if show_probs and hasattr(model["rf"], "predict_proba"):
            probs = model["rf"].predict_proba(new_imp)
            for i, cls in enumerate(model["rf"].classes_):
                out[f"proba_{cls}"] = probs[:, i]

        st.write("**Predictions (preview)**")
        st.dataframe(out.head())

        buf = io.StringIO()
        out.to_csv(buf, index=False)
        st.download_button("Download predictions.csv", buf.getvalue(), "predictions.csv", "text/csv")

# Footer hint
st.caption("Tip: For Kepler KOI, try `koi_pdisposition` as the label. For TESS, try `disposition` or `tfopwg_disp`.")

