# Heart Disease Prediction System

Machine learning project predicting the presence of heart disease from clinical patient data, using the combined UCI Heart Disease dataset (Cleveland, Hungary, Switzerland, VA Long Beach — 920 patients).

## Project status

- Phase 1 — EDA & Preprocessing
- Phase 2 — Exploratory data analysis
- Phase 3 — Model training & tuning (XGBoost)
- Phase 4 — Interpretability (SHAP)
- Phase 5 — Deployment (Streamlit)
- Phase 6 — Documentation

## Dataset

Combined UCI Heart Disease dataset (920 patients, 4 hospitals). Missingness is structural (hospital-dependent) rather than random, and `chol`/`trestbps` contain literal `0` values representing unmeasured data — both handled explicitly during preprocessing. See `notebooks/01_eda_preprocessing.ipynb` for details.

Target: binary classification — presence (1) vs absence (0) of heart disease.

### Input features

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | Sex (Male / Female) |
| `cp` | Chest pain type: typical angina, atypical angina, non-anginal, asymptomatic |
| `trestbps` | Resting blood pressure (mmHg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (True/False) |
| `restecg` | Resting ECG results: normal, lv hypertrophy, st-t abnormality |
| `thalch` | Maximum heart rate achieved during a stress test |
| `exang` | Exercise-induced angina (True/False) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment: upsloping, flat, downsloping |
| `ca` | Number of major vessels (0–3) colored by fluoroscopy |
| `thal` | Thalassemia test result: normal, fixed defect, reversable defect |

## Model & results

**Model**: XGBoost, tuned via `RandomizedSearchCV` (50 iterations, 5-fold stratified cross-validation, optimized for F1-score).

**Test set performance**:

| Metric | Score |
|---|---|
| Accuracy | 0.85 |
| F1-score (Disease) | 0.87 |
| Recall (Disease) | 0.90 |
| ROC-AUC | 0.91 |

**Top predictive features** (SHAP): `ca` (number of major vessels), `exang` (exercise-induced angina), `cp_atypical angina`, `sex`, `oldpeak`.

See `notebooks/04_interpretability_shap.ipynb` for global and per-patient SHAP explanations.

## Repository structure
├── data/
│ ├── raw/ # original dataset
│ └── processed/ # train/test splits after preprocessing
├── notebooks/ # numbered, ordered analysis notebooks
├── src/ # reusable preprocessing/training/prediction code
├── models/ # serialized trained model
├── app/ # Streamlit demo app
├── reports/figures/ # exported plots
└── docs/ # project spec (cahier des charges)


## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the notebooks in order (`notebooks/01_...` through `04_...`) to reproduce the full pipeline, or run the Streamlit app once available:

```bash
streamlit run app/streamlit_app.py
```

## Tech stack

Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit.
