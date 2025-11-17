import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Parkinson's Disease Detection", layout="wide")

st.markdown("""
    <style>
    .metric-positive {background-color: #ffebee; padding: 20px; border-radius: 10px; border-left: 5px solid #f44336;}
    .metric-negative {background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #4caf50;}
    </style>
""", unsafe_allow_html=True)

FEATURE_COLUMNS = [
    'Age', 'Gender', 'BMI', 'Smoking', 'AlcoholConsumption', 'PhysicalActivity',
    'DietQuality', 'SleepQuality', 'FamilyHistoryParkinsons', 'TraumaticBrainInjury',
    'Hypertension', 'Diabetes', 'Depression', 'Stroke', 'SystolicBP', 'DiastolicBP',
    'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides',
    'MoCA', 'FunctionalAssessment', 'Tremor', 'Rigidity', 'Bradykinesia',
    'PosturalInstability', 'SpeechProblems', 'SleepDisorders', 'Constipation'
]

@st.cache_resource
def train_model():
    df = pd.read_csv('parkinsons_disease_data.csv')
    X = df[FEATURE_COLUMNS]
    y = df['Diagnosis']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    accuracy = clf.score(X_test, y_test)
    return clf, accuracy

clf_model, accuracy = train_model()

st.title("🧠 Parkinson's Disease Detection")
st.metric("Model Accuracy", f"{accuracy:.2%}")
st.markdown("---")

st.sidebar.header("📋 Patient Information")

st.sidebar.subheader("Demographics")
age = st.sidebar.slider("Age (years)", 18, 100, 65)
gender = st.sidebar.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
bmi = st.sidebar.slider("BMI", 10.0, 60.0, 24.0, 0.1)

st.sidebar.subheader("Lifestyle")
smoking = st.sidebar.selectbox("Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
alcohol = st.sidebar.slider("Alcohol", 0.0, 10.0, 2.0, 0.1)
physical_activity = st.sidebar.slider("Physical Activity", 0.0, 10.0, 7.0, 0.1)
diet_quality = st.sidebar.slider("Diet Quality", 0.0, 10.0, 7.0, 0.1)
sleep_quality = st.sidebar.slider("Sleep Quality", 0.0, 10.0, 7.0, 0.1)

st.sidebar.subheader("Medical History")
family_history = st.sidebar.selectbox("Family History", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
traumatic_brain_injury = st.sidebar.selectbox("Brain Injury", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
hypertension = st.sidebar.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
diabetes = st.sidebar.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
depression = st.sidebar.selectbox("Depression", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
stroke = st.sidebar.selectbox("Stroke", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

st.sidebar.subheader("Clinical Measurements")
systolic_bp = st.sidebar.slider("Systolic BP", 80, 200, 120)
diastolic_bp = st.sidebar.slider("Diastolic BP", 40, 130, 80)
cholesterol_total = st.sidebar.slider("Total Cholesterol", 100, 400, 180)
cholesterol_ldl = st.sidebar.slider("LDL", 50, 250, 100)
cholesterol_hdl = st.sidebar.slider("HDL", 20, 100, 55)
cholesterol_triglycerides = st.sidebar.slider("Triglycerides", 50, 400, 100)

st.sidebar.subheader("Cognitive & Functional")
moca = st.sidebar.slider("MoCA Score", 0, 30, 26)
functional_assessment = st.sidebar.slider("Functional Assessment", 0.0, 10.0, 8.0, 0.1)

st.sidebar.subheader("Symptoms")
tremor = st.sidebar.slider("Tremor (0-10)", 0, 10, 0)
rigidity = st.sidebar.slider("Rigidity (0-10)", 0, 10, 0)
bradykinesia = st.sidebar.slider("Bradykinesia (0-10)", 0, 10, 0)
postural_instability = st.sidebar.slider("Postural Instability (0-10)", 0, 10, 0)
speech_problems = st.sidebar.selectbox("Speech Problems", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
sleep_disorders = st.sidebar.selectbox("Sleep Disorders", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
constipation = st.sidebar.selectbox("Constipation", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

predict_btn = st.sidebar.button("🔍 Predict Diagnosis", use_container_width=True)

if predict_btn:
    input_data = pd.DataFrame({
        'Age': [age], 'Gender': [gender], 'BMI': [bmi], 'Smoking': [smoking],
        'AlcoholConsumption': [alcohol], 'PhysicalActivity': [physical_activity],
        'DietQuality': [diet_quality], 'SleepQuality': [sleep_quality],
        'FamilyHistoryParkinsons': [family_history], 'TraumaticBrainInjury': [traumatic_brain_injury],
        'Hypertension': [hypertension], 'Diabetes': [diabetes], 'Depression': [depression],
        'Stroke': [stroke], 'SystolicBP': [systolic_bp], 'DiastolicBP': [diastolic_bp],
        'CholesterolTotal': [cholesterol_total], 'CholesterolLDL': [cholesterol_ldl],
        'CholesterolHDL': [cholesterol_hdl], 'CholesterolTriglycerides': [cholesterol_triglycerides],
        'MoCA': [moca], 'FunctionalAssessment': [functional_assessment],
        'Tremor': [tremor], 'Rigidity': [rigidity], 'Bradykinesia': [bradykinesia],
        'PosturalInstability': [postural_instability], 'SpeechProblems': [speech_problems],
        'SleepDisorders': [sleep_disorders], 'Constipation': [constipation]
    })[FEATURE_COLUMNS]
    
    diagnosis_pred = clf_model.predict(input_data)[0]
    diagnosis_prob = clf_model.predict_proba(input_data)[0]
    
    st.header("📊 Prediction Results")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if diagnosis_pred == 1:
            st.markdown("""
            <div class="metric-positive">
                <h2>🔴 Parkinson's Disease: <span style='color: #f44336;'>POSITIVE</span></h2>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-negative">
                <h2>🟢 Parkinson's Disease: <span style='color: #4caf50;'>NEGATIVE</span></h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Negative %", f"{diagnosis_prob[0]*100:.1f}%")
        st.metric("Positive %", f"{diagnosis_prob[1]*100:.1f}%")
    
    st.markdown("---")
    
    feature_importance = clf_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': FEATURE_COLUMNS,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False).head(10)
    
    st.subheader("Top Features")
    st.bar_chart(importance_df.set_index('Feature'))
    
    results_df = pd.DataFrame({
        'Metric': ['Diagnosis', 'Negative %', 'Positive %'],
        'Value': ['POSITIVE' if diagnosis_pred == 1 else 'NEGATIVE', 
                  f"{diagnosis_prob[0]*100:.1f}", f"{diagnosis_prob[1]*100:.1f}"]
    })
    
    st.download_button("📥 Download", results_df.to_csv(index=False), "prediction.csv", "text/csv")

else:
    st.info("👈 Enter patient info and click Predict")

st.caption("⚠️ For research only. Not for clinical diagnosis.")

only change the UI of my project,make it super elegant easy to sue and attractive , no need to interfere in my ML modelse or anythin, just UI, change only HTML, CSS, no change in input feautres nad models is allowed here
