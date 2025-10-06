import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

st.set_page_config(page_title="ExoFindr — Exoplanet Classifier", layout="wide")
st.title("ExoFindr — Exoplanet Classifier (Kepler / K2 / TESS)")
st.caption("NASA Space Apps 2025 — Interactive AI Model to classify exoplanets.")

# Load your pre-uploaded dataset directly from the repo
@st.cache_data
def load_data():
    return pd.read_csv("kepler_cumulative.csv", comment="#", low_memory=False)

df = load_data()
st.success("Kepler cumulative dataset loaded successfully!")

# Preprocess + train model once
NUMERIC_COLS = ["koi_period", "koi_duration", "koi_depth", "koi_prad",
                "koi_teq", "koi_insol", "koi_model_snr", "koi_impact",
                "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag"]

df = df.dropna(subset=["koi_disposition"])
df["koi_disposition"] = df["koi_disposition"].str.upper().replace({
    "PC": "CANDIDATE", "FP": "FALSE POSITIVE"
})

X = df[NUMERIC_COLS]
y = df["koi_disposition"]

imp = SimpleImputer(strategy="median")
X = imp.fit_transform(X)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)

# Manual input interface
st.subheader("🔭 Try it yourself — Manually enter exoplanet data")
inputs = {}
cols = st.columns(3)
for i, feat in enumerate(NUMERIC_COLS):
    inputs[feat] = cols[i % 3].number_input(feat, value=float(np.median(df[feat])))

if st.button("Predict Exoplanet Type"):
    test_df = pd.DataFrame([inputs])
    test_df = imp.transform(test_df)
    pred = rf.predict(test_df)[0]
    st.success(f"The model predicts: **{pred}**")
