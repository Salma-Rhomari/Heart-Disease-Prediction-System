import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import plotly.graph_objects as go

st.set_page_config(page_title="Heart Disease Prediction", page_icon="", layout="wide")

PRIMARY = "#1E5AA8"     # deep blue
LIGHT = "#5B9BD5"       # mid blue
PALE = "#DCEBFA"        # pale blue background
DARK = "#0B2E52"        # near-navy text/dark accents

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


@st.cache_resource
def load_model():
    return joblib.load('models/model.pkl')


@st.cache_data
def load_reference_data():
    return pd.read_csv('data/processed/X_train.csv')


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


model = load_model()
X_train = load_reference_data()
X_train_columns = X_train.columns.tolist()
explainer = load_explainer(model)


def plot_shap_waterfall_blue(shap_values_row, feature_names, base_value, prediction_value, max_display=10):
    """Custom blue-only waterfall plot for a single SHAP explanation."""
    values = shap_values_row
    order = np.argsort(np.abs(values))[::-1][:max_display]

    names = [feature_names[i] for i in order]
    vals = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    y_pos = np.arange(len(names))

    colors = [PRIMARY if v > 0 else LIGHT for v in vals]
    ax.barh(y_pos, vals, color=colors, height=0.6)

    for i, v in enumerate(vals):
        ax.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:+.2f}",
                va='center', ha='left' if v >= 0 else 'right',
                fontsize=9, color=DARK, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(0, color=DARK, linewidth=0.8)
    ax.set_xlabel("Impact on prediction (SHAP value)")
    ax.set_title(f"Base rate: {base_value:.2f}  →  This patient: {prediction_value:.2f}",
                 fontsize=10, color=DARK)

    legend_elements = [
        Patch(facecolor=PRIMARY, label='Pushes toward Disease'),
        Patch(facecolor=LIGHT, label='Pushes toward Healthy'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


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

st.markdown(f"<h1 style='color:{DARK};'> Heart Disease Prediction System</h1>", unsafe_allow_html=True)
st.markdown("Enter the patient's clinical parameters below to predict the likelihood of heart disease, with a full explanation of the prediction.")
st.divider()

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

if st.button(" Predict", type="primary", use_container_width=True):
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
                <h2 style='color:white; margin:0;'> No Disease Likely</h2>
                <p style='color:white; font-size:1.1rem;'>Predicted probability of disease: {proba:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

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


    st.divider()
    st.subheader(" Why this prediction? (SHAP explanation)")
    st.markdown("Each bar shows how much a feature pushed the prediction toward **Disease** (dark blue) or **Healthy** (light blue) for this specific patient.")

    shap_values_patient = explainer(row)

    fig = plot_shap_waterfall_blue(
        shap_values_row=shap_values_patient.values[0],
        feature_names=X_train_columns,
        base_value=shap_values_patient.base_values[0],
        prediction_value=shap_values_patient.base_values[0] + shap_values_patient.values[0].sum(),
        max_display=10
    )
    st.pyplot(fig)
    age = st.number_input(
    "Age",
    min_value=18, max_value=100,
    help="Patient's age in years"
)

cp = st.selectbox(
    "Chest Pain Type",
    options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
    help="Type of chest pain experienced by the patient"
)

thalach = st.number_input(
    "Max Heart Rate Achieved",
    min_value=60, max_value=220,
    help="Maximum heart rate achieved during exercise test"
)