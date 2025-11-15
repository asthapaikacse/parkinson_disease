import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def load_model():
    return joblib.load('pd_classifier_model.pkl')

model = load_model()

# DO NOT include 'Ethnicity' or 'EducationLevel' in the UI or input!
FEATURES = [
    'Age', 'Gender', 'BMI', 'Smoking',
    'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
    'FamilyHistoryParkinsons', 'TraumaticBrainInjury', 'Hypertension', 'Diabetes',
    'Depression', 'Stroke', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal',
    'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'MoCA',
    'FunctionalAssessment', 'Tremor', 'Rigidity', 'Bradykinesia',
    'PosturalInstability', 'SpeechProblems', 'SleepDisorders', 'Constipation'
]

CAT_FEATURES = ['Gender', 'Smoking']

st.title("Parkinson's Disease Detection")

with st.form("pd_form"):
    st.write("Enter patient details below:")
    patient_input = {}
    for feat in FEATURES:
        if feat in CAT_FEATURES:
            if feat == 'Gender':
                val = st.selectbox(feat, [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
            elif feat == 'Smoking':
                val = st.selectbox(feat, [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        else:
            val = st.number_input(feat, value=0.0)
        patient_input[feat] = val
    submitted = st.form_submit_button("Predict")

if submitted:
    # You must include columns in the same order/length as your model ("Ethnicity" and "EducationLevel" can be filled as most common or 0)
    df_input = pd.DataFrame([patient_input])
    # Fill dropped features with dummy value (0) to match model's expected columns
    for col in ['Ethnicity', 'EducationLevel']:
        df_input[col] = 0
    # Reorder columns to match model input
    model_cols = [  # Same as training order!
        'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'Smoking',
        'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
        'FamilyHistoryParkinsons', 'TraumaticBrainInjury', 'Hypertension', 'Diabetes',
        'Depression', 'Stroke', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal',
        'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'MoCA',
        'FunctionalAssessment', 'Tremor', 'Rigidity', 'Bradykinesia',
        'PosturalInstability', 'SpeechProblems', 'SleepDisorders', 'Constipation'
    ]
    df_input = df_input.reindex(columns=model_cols)
    pred = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]
    st.markdown("---")
    if pred == 1:
        st.error(f"🔴 Likely Parkinson's Disease (confidence: {prob*100:.1f}%)")
    else:
        st.success(f"🟢 No Parkinson's Disease detected (confidence: {(1-prob)*100:.1f}%)")
    st.markdown("---")
    st.write("**Input Features:**")
    st.json(patient_input)
