# ExoFindr — Kepler Exoplanet Classifier (NASA Space Apps 2025)

This repo contains a minimal, submit-ready package to train and demo an ML model that classifies **planetary candidates vs. false positives** using the **Kepler KOI cumulative** dataset.

## 🔧 What’s included
- `train_evaluate.py` — trains a RandomForest on the KOI CSV (skipping comment lines), prints Accuracy/F1, saves a model.
- `app.py` — Streamlit app to upload a KOI CSV, train, view metrics/plots, and predict on new rows.
- `requirements.txt` — Python dependencies.

## ▶️ Quickstart
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train + evaluate from a local KOI CSV
python train_evaluate.py --csv cumulative.csv --target koi_pdisposition

# Launch the web app
streamlit run app.py
```

## 🗂️ Data (Kepler KOI)
Use the Kepler cumulative KOI table CSV (comma-separated, with header after comment lines). The label column is typically **`koi_pdisposition`** (Disposition Using Kepler Data).

## 📜 License
MIT
