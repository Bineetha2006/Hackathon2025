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


st.markdown("""
### 🌍 What Your Prediction Means
**CONFIRMED:** The data strongly indicates that this object is a verified exoplanet — real and validated by multiple observations.  
**CANDIDATE:** The signal looks promising, but further observation is needed to confirm if it's truly an exoplanet.  
**FALSE POSITIVE:** The data pattern likely results from a star, noise, or another non-planetary object.

### 🧾 Explanation of Input Fields
These are the parameters your prediction is based on:
- **koi_period** — Time the planet takes to orbit its star (in days).  
- **koi_duration** — Duration of the transit (in hours).  
- **koi_depth** — How much the star’s brightness dips when the planet passes in front (in parts per million).  
- **koi_prad** — Estimated planet radius (in Earth radii).  
- **koi_teq** — Planet’s equilibrium temperature (in Kelvin).  
- **koi_insol** — Amount of stellar energy the planet receives (relative to Earth).  
- **koi_model_snr** — Signal-to-noise ratio of the detection (higher = more reliable).  
- **koi_impact** — How central the transit path is across the star (0 = perfect center).  
- **koi_steff** — The star’s surface temperature (Kelvin).  
- **koi_slogg** — Star’s surface gravity.  
- **koi_srad** — Star’s radius (in solar radii).  
- **koi_kepmag** — Kepler brightness magnitude (smaller = brighter).

✨ Your inputs help the AI model decide if the pattern matches that of a real planet or not.
""")

