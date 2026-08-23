# Heart Disease Prediction System

Machine learning project predicting the presence of heart disease from clinical patient data, using the combined UCI Heart Disease dataset (Cleveland, Hungary, Switzerland, VA Long Beach — 920 patients).

## Project status

- [x] Phase 1 — EDA & Preprocessing
- [ ] Phase 2 — Exploratory data analysis
- [ ] Phase 3 — Model comparison & tuning
- [ ] Phase 4 — Interpretability (SHAP)
- [ ] Phase 5 — Deployment (Streamlit)
- [ ] Phase 6 — Documentation

## Dataset

Combined UCI Heart Disease dataset (920 patients, 4 hospitals). Missingness is structural (hospital-dependent) rather than random — see `notebooks/01_eda_preprocessing.ipynb` for details.

Target: binary classification — presence (1) vs absence (0) of heart disease.

## Repository structure

```
├── data/
│   ├── raw/                # original dataset
│   └── processed/          # train/test splits after preprocessing
├── notebooks/               # numbered, ordered analysis notebooks
├── src/                      # reusable preprocessing/training/prediction code
├── models/                  # serialized trained model
├── app/                      # Streamlit demo app
├── reports/figures/         # exported plots
└── docs/                     # project spec (cahier des charges)
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the notebooks in order (`notebooks/01_...` through `04_...`) to reproduce the full pipeline, or run the Streamlit app once the model is trained:

```bash
streamlit run app/streamlit_app.py
```

## Tech stack

Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit.

## License

MIT
