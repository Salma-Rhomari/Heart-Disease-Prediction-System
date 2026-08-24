import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go

# ---------- Page config & style ----------
st.set_page_config(page_title="Heart Disease Prediction", page_icon="", layout="wide")

PRIMARY = "#1E5AA8"     
LIGHT = "#5B9BD5"       
PALE = "#DCEBFA"        
DARK = "#0B2E52"        

st.markdown(f"""
<style>
    .stApp {{ background-color: #F5F9FE; }}
    h1, h2, h3 {{ color: {DARK}; }}
    .stButton>button {{
        background-color: {PRIMARY};
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
    }}
    .stButton>button:hover {{
        background-color: {DARK};
        color: white;
    }}
    div[data-testid="stMetricValue"] {{ color: {PRIMARY}; }}
    section[data-testid="stSidebar"] {{ background-color: {PALE}; }}
</style>
""", unsafe_allow_html=True)

# ---------- Load model & data ----------
@st.cache_resource
def load_model():
    return joblib.load('models/model.pkl')

@st.cache_data
def load_reference_data():
    X_train = pd.read_csv('data/processed/X_train.csv')
    return X_train

model = load_model()
X_train = load_reference_data()
X_train_columns = X_train.columns.tolist()

@st.cache_resource
def load_explainer():
    return shap.TreeExplainer(model)

explainer = load_explainer()

# ---------- Sidebar: model info ----------
with st.sidebar:
    st.markdown(f"<h2 style='color:{DARK};'> Model Info</h2>", unsafe_allow_html=True)
    st.markdown("**Algorithm:** XGBoost (tuned)")
    st.markdown("**Dataset:** UCI Heart Disease — combined (Cleveland, Hungary, Switzerland, VA Long Beach), 920 patients")

    st.markdown("---")
    st.markdown("**Test set performance**")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", "85%")
    c2.metric("ROC-AUC", "0.91")
    c1.metric("F1 (Disease)", "0.87")
    c2.metric("Recall (Disease)", "0.90")

    st.markdown("---")
    st.markdown("**Top predictors (SHAP)**")
    st.markdown("`ca` · `exang` · `cp_atypical angina` · `sex` · `oldpeak`")

    st.markdown("---")
    st.caption(" Educational/portfolio project — not a medical diagnosis tool.")

# ---------- Main title ----------
st.markdown(f"<h1 style='color:{DARK};'> Heart Disease Prediction System</h1>", unsafe_allow_html=True)
st.markdown("Enter the patient's clinical parameters below to predict the likelihood of heart disease, with a full explanation of the prediction.")
st.divider()

# ---------- Input form ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Profile")
    age = st.number_input("Age", min_value=1, max_value=120, value=50,
                           help="Patient's age in years")
    sex = st.selectbox("Sex", ["Male", "Female"],
                        help="Biological sex of the patient")
    cp = st.selectbox("Chest pain type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
                       help="Type of chest pain experienced")
    trestbps = st.number_input("Resting blood pressure (mmHg)", min_value=50, max_value=250, value=130,
                                help="Blood pressure measured at rest")
    chol = st.number_input("Cholesterol (mg/dl)", min_value=50, max_value=700, value=230,
                            help="Serum cholesterol level")
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl?", ["False", "True"],
                        help="Whether fasting blood sugar exceeds 120 mg/dl (diabetes indicator)")
    restecg = st.selectbox("Resting ECG result", ["normal", "lv hypertrophy", "st-t abnormality"],
                            help="Result of the resting electrocardiogram")

with col2:
    st.subheader("Exercise Test Results")
    thalch = st.number_input("Max heart rate achieved", min_value=50, max_value=250, value=150,
                              help="Maximum heart rate reached during a stress test")
    exang = st.selectbox("Exercise-induced angina?", ["False", "True"],
                          help="Chest pain triggered specifically by physical exercise")
    oldpeak = st.number_input("ST depression (oldpeak)", min_value=-3.0, max_value=7.0, value=1.0, step=0.1,
                               help="ST segment depression induced by exercise relative to rest")
    slope = st.selectbox("Slope of peak exercise ST segment", ["upsloping", "flat", "downsloping"],
                          help="Shape of the ST segment during peak exercise")
    ca = st.number_input("Number of major vessels (0-3)", min_value=0, max_value=3, value=0,
                          help="Number of major blood vessels colored by fluoroscopy")
    thal = st.selectbox("Thalassemia test result", ["normal", "fixed defect", "reversable defect"],
                         help="Result of the thalassemia blood disorder test")

st.divider()

# ---------- Prediction ----------
if st.button("🔍 Predict", type="primary", use_container_width=True):
    input_dict = {
        'id': 0,
        'age': age,
        'sex': 1 if sex == "Male" else 0,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': 1 if fbs == "True" else 0,
        'thalch': thalch,
        'exang': 1 if exang == "True" else 0,
        'oldpeak': oldpeak,
        'ca': ca,
        'ca_missing': 0,
        'thal_missing': 0,
        'slope_missing': 0,
    }
    row = pd.DataFrame([input_dict])

    for col in ['cp_atypical angina', 'cp_non-anginal', 'cp_typical angina']:
        row[col] = 1 if col == f'cp_{cp}' else 0
    for col in ['restecg_normal', 'restecg_st-t abnormality']:
        row[col] = 1 if col == f'restecg_{restecg}' else 0
    for col in ['slope_flat', 'slope_upsloping']:
        row[col] = 1 if col == f'slope_{slope}' else 0
    for col in ['thal_normal', 'thal_reversable defect']:
        row[col] = 1 if col == f'thal_{thal}' else 0

    row = row.reindex(columns=X_train_columns, fill_value=0)

    prediction = model.predict(row)[0]
    proba = model.predict_proba(row)[0][1]

    st.divider()

    # ----- Result banner -----
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        if prediction == 1:
            st.markdown(f"""
            <div style='background-color:{PRIMARY}; padding:1.5rem; border-radius:12px; text-align:center;'>
                <h2 style='color:white; margin:0;'> Disease Likely</h2>
                <p style='color:white; font-size:1.1rem;'>Predicted probability: {proba:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background-color:{LIGHT}; padding:1.5rem; border-radius:12px; text-align:center;'>
                <h2 style='color:white; margin:0;'>
                 
             No Disease Likely</h2>
                <p style='color:white; font-size:1.1rem;'>Predicted probability of disease: {proba:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

    # ----- Gauge chart -----
    with res_col2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={'suffix': "%", 'font': {'color': DARK}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': DARK},
                'bar': {'color': PRIMARY},
                'bgcolor': "white",
                'steps': [
                    {'range': [0, 40], 'color': PALE},
                    {'range': [40, 70], 'color': LIGHT},
                    {'range': [70, 100], 'color': PRIMARY},
                ],
            },
            title={'text': "Disease Probability", 'font': {'color': DARK, 'size': 16}}
        ))
        fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.caption("This tool is for educational/portfolio purposes only and is not a medical diagnosis.")

    # ----- SHAP explanation for this patient -----
    st.divider()
    st.subheader("🔎 Why this prediction? (SHAP explanation)")
    st.markdown("Each bar shows how much a feature pushed the prediction toward **Disease** (blue, right) or **Healthy** (light blue, left) for this specific patient.")

    shap_values_patient = explainer(row)

    blue_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue_scale", [LIGHT, PRIMARY])

    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(shap_values_patient[0], show=False)
    for fc in ax.get_children():
        pass  # waterfall colors are managed internally; see note below
    plt.tight_layout()
    st.pyplot(fig)