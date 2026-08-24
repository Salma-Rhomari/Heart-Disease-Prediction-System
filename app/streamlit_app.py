import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Prediction", page_icon="", layout="centered")

model = joblib.load('models/model.pkl')
X_train_columns = pd.read_csv('data/processed/X_train.csv').columns.tolist()

st.title(" Heart Disease Prediction System")
st.markdown("Enter the patient's clinical parameters below to predict the likelihood of heart disease.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50,
                           help="Patient's age in years")
    sex = st.selectbox("Sex", ["Male", "Female"],
                        help="Biological sex of the patient")
    cp = st.selectbox("Chest pain type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
                       help="Type of chest pain experienced")
    trestbps = st.number_input("Resting blood pressure (mmHg)", min_value=50, max_value=250, value=130,
                                help="Blood pressure at rest")
    chol = st.number_input("Cholesterol (mg/dl)", min_value=50, max_value=700, value=230,
                            help="Serum cholesterol level")
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl?", ["False", "True"],
                        help="Whether fasting blood sugar exceeds 120 mg/dl (diabetes indicator)")
    restecg = st.selectbox("Resting ECG result", ["normal", "lv hypertrophy", "st-t abnormality"],
                            help="Result of the resting electrocardiogram")

with col2:
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

if st.button("Predict", type="primary", use_container_width=True):
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

    for cat_col, cat_val, prefix in [
        ('cp', cp, 'cp'), ('restecg', restecg, 'restecg'),
        ('slope', slope, 'slope'), ('thal', thal, 'thal')
    ]:
        pass

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
    if prediction == 1:
        st.error(f" **Disease likely** — predicted probability: {proba:.1%}")
    else:
        st.success(f" **No disease likely** — predicted probability of disease: {proba:.1%}")

    st.caption("This tool is for educational/portfolio purposes only and is not a medical diagnosis.")