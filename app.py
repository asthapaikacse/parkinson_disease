import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split

# Page config
st.set_page_config(page_title="Parkinson's Disease Detection", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 12px 24px;
    }
    .metric-positive {
        background-color: #ffebee;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #f44336;
    }
    .metric-negative {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

# Define feature columns (29 features after dropping PatientID, DoctorInCharge, Ethnicity, EducationLevel, Diagnosis, UPDRS)
FEATURE_COLUMNS = [
    'Age', 'Gender', 'BMI', 'Smoking', 'AlcoholConsumption', 'PhysicalActivity',
    'DietQuality', 'SleepQuality', 'FamilyHistoryParkinsons', 'TraumaticBrainInjury',
    'Hypertension', 'Diabetes', 'Depression', 'Stroke', 'SystolicBP', 'DiastolicBP',
    'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides',
    'MoCA', 'FunctionalAssessment', 'Tremor', 'Rigidity', 'Bradykinesia',
    'PosturalInstability', 'SpeechProblems', 'SleepDisorders', 'Constipation'
]

@st.cache_resource
def load_and_train_models():
    """Load data and train CatBoost models (cached)"""
    
    # Load data
    df = pd.read_csv('parkinsons_disease_data.csv')
    
    # Prepare features and targets
    X = df[FEATURE_COLUMNS]
    y_diagnosis = df['Diagnosis']
    y_updrs = df['UPDRS']
    
    # Train-test split
    X_train, X_test, y_train_diag, y_test_diag = train_test_split(
        X, y_diagnosis, test_size=0.2, random_state=42, stratify=y_diagnosis
    )
    
    _, _, y_train_updrs, y_test_updrs = train_test_split(
        X, y_updrs, test_size=0.2, random_state=42
    )
    
    # Train CatBoost Classifier (for Diagnosis)
    clf = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        random_seed=42,
        verbose=False,
        auto_class_weights='Balanced'
    )
    clf.fit(X_train, y_train_diag)
    
    # Train CatBoost Regressor (for UPDRS)
    reg = CatBoostRegressor(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        random_seed=42,
        verbose=False
    )
    reg.fit(X_train, y_train_updrs)
    
    # Calculate accuracies
    clf_accuracy = clf.score(X_test, y_test_diag)
    reg_score = reg.score(X_test, y_test_updrs)
    
    return clf, reg, clf_accuracy, reg_score

# Load models
with st.spinner("🔄 Training models (first run only)..."):
    clf_model, reg_model, clf_acc, reg_r2 = load_and_train_models()

# Title
st.title("🧠 Parkinson's Disease Detection & Severity Prediction")
st.markdown("**ML-Based Clinical Decision Support System**")

# Model performance
col1, col2 = st.columns(2)
with col1:
    st.metric("Classification Model Accuracy", f"{clf_acc:.2%}")
with col2:
    st.metric("Regression Model R² Score", f"{reg_r2:.3f}")

st.markdown("---")

# Sidebar inputs
st.sidebar.header("📋 Patient Information")

# Demographics
st.sidebar.subheader("Demographics")
age = st.sidebar.slider("Age (years)", 18, 100, 60)
gender = st.sidebar.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
bmi = st.sidebar.slider("BMI", 10.0, 60.0, 25.0, 0.1)

# Lifestyle
st.sidebar.subheader("Lifestyle Factors")
smoking = st.sidebar.selectbox("Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
alcohol = st.sidebar.slider("Alcohol Consumption", 0.0, 10.0, 5.0, 0.1)
physical_activity = st.sidebar.slider("Physical Activity", 0.0, 10.0, 5.0, 0.1)
diet_quality = st.sidebar.slider("Diet Quality", 0.0, 10.0, 5.0, 0.1)
sleep_quality = st.sidebar.slider("Sleep Quality", 0.0, 10.0, 5.0, 0.1)

# Medical History
st.sidebar.subheader("Medical History")
family_history = st.sidebar.selectbox("Family History of Parkinson's", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
traumatic_brain_injury = st.sidebar.selectbox("Traumatic Brain Injury", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
hypertension = st.sidebar.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
diabetes = st.sidebar.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
depression = st.sidebar.selectbox("Depression", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
stroke = st.sidebar.selectbox("Stroke", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

# Clinical Measurements
st.sidebar.subheader("Clinical Measurements")
systolic_bp = st.sidebar.slider("Systolic BP (mmHg)", 80, 200, 120)
diastolic_bp = st.sidebar.slider("Diastolic BP (mmHg)", 40, 130, 80)
cholesterol_total = st.sidebar.slider("Total Cholesterol", 100, 400, 200)
cholesterol_ldl = st.sidebar.slider("LDL Cholesterol", 50, 250, 120)
cholesterol_hdl = st.sidebar.slider("HDL Cholesterol", 20, 100, 50)
cholesterol_triglycerides = st.sidebar.slider("Triglycerides", 50, 400, 150)

# Cognitive & Functional
st.sidebar.subheader("Cognitive & Functional")
moca = st.sidebar.slider("MoCA Score", 0, 30, 20)
functional_assessment = st.sidebar.slider("Functional Assessment", 0.0, 10.0, 5.0, 0.1)

# Symptoms
st.sidebar.subheader("Symptoms")
tremor = st.sidebar.slider("Tremor (0-10)", 0, 10, 5)
rigidity = st.sidebar.slider("Rigidity (0-10)", 0, 10, 5)
bradykinesia = st.sidebar.slider("Bradykinesia (0-10)", 0, 10, 5)
postural_instability = st.sidebar.slider("Postural Instability (0-10)", 0, 10, 5)
speech_problems = st.sidebar.selectbox("Speech Problems", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
sleep_disorders = st.sidebar.selectbox("Sleep Disorders", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
constipation = st.sidebar.selectbox("Constipation", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

# Predict button
predict_btn = st.sidebar.button("🔍 Predict Diagnosis", use_container_width=True)

# Main content
if predict_btn:
    # Create input dataframe
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'BMI': [bmi],
        'Smoking': [smoking],
        'AlcoholConsumption': [alcohol],
        'PhysicalActivity': [physical_activity],
        'DietQuality': [diet_quality],
        'SleepQuality': [sleep_quality],
        'FamilyHistoryParkinsons': [family_history],
        'TraumaticBrainInjury': [traumatic_brain_injury],
        'Hypertension': [hypertension],
        'Diabetes': [diabetes],
        'Depression': [depression],
        'Stroke': [stroke],
        'SystolicBP': [systolic_bp],
        'DiastolicBP': [diastolic_bp],
        'CholesterolTotal': [cholesterol_total],
        'CholesterolLDL': [cholesterol_ldl],
        'CholesterolHDL': [cholesterol_hdl],
        'CholesterolTriglycerides': [cholesterol_triglycerides],
        'MoCA': [moca],
        'FunctionalAssessment': [functional_assessment],
        'Tremor': [tremor],
        'Rigidity': [rigidity],
        'Bradykinesia': [bradykinesia],
        'PosturalInstability': [postural_instability],
        'SpeechProblems': [speech_problems],
        'SleepDisorders': [sleep_disorders],
        'Constipation': [constipation]
    })
    
    # Ensure column order matches training
    input_data = input_data[FEATURE_COLUMNS]
    
    # Make predictions
    diagnosis_pred = clf_model.predict(input_data)[0]
    diagnosis_prob = clf_model.predict_proba(input_data)[0]
    updrs_pred = reg_model.predict(input_data)[0]
    
    # Display results
    st.header("📊 Prediction Results")
    
    # Diagnosis result
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if diagnosis_pred == 1:
            st.markdown(f"""
            <div class="metric-positive">
                <h2>🔴 Parkinson's Disease: <span style='color: #f44336;'>POSITIVE</span></h2>
                <p style='font-size: 18px;'>The model predicts that the patient likely has Parkinson's disease.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-negative">
                <h2>🟢 Parkinson's Disease: <span style='color: #4caf50;'>NEGATIVE</span></h2>
                <p style='font-size: 18px;'>The model predicts that the patient likely does NOT have Parkinson's disease.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Confidence (Negative)", f"{diagnosis_prob[0]*100:.1f}%")
        st.metric("Confidence (Positive)", f"{diagnosis_prob[1]*100:.1f}%")
    
    st.markdown("---")
    
    # UPDRS & Severity
    st.subheader("📈 Severity Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Predicted UPDRS Score", f"{updrs_pred:.1f}")
        st.caption("Range: 0-176 (higher = more severe)")
    
    with col2:
        # ML-based severity categorization (quantile-based)
        if updrs_pred < 25:
            severity = "🟢 Mild"
            severity_desc = "Early stage, minimal motor impairment"
        elif updrs_pred < 50:
            severity = "🟡 Moderate"
            severity_desc = "Moderate symptoms, some functional limitation"
        else:
            severity = "🔴 Severe"
            severity_desc = "Advanced stage, significant impairment"
        
        st.metric("Severity Category", severity)
        st.caption(severity_desc)
    
    with col3:
        # Risk score (normalized)
        risk_score = min(100, (diagnosis_prob[1] * 100 + updrs_pred / 1.76) / 2)
        st.metric("Overall Risk Score", f"{risk_score:.1f}/100")
        st.caption("Combined diagnosis probability and severity")
    
    st.markdown("---")
    
    # Feature importance
    st.subheader("🔍 Feature Importance (Top 10)")
    feature_importance = clf_model.get_feature_importance()
    importance_df = pd.DataFrame({
        'Feature': FEATURE_COLUMNS,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False).head(10)
    
    st.bar_chart(importance_df.set_index('Feature'))
    
    st.markdown("---")
    
    # Download results
    results_df = pd.DataFrame({
        'Metric': ['Diagnosis', 'Confidence (Negative)', 'Confidence (Positive)', 'UPDRS Score', 'Severity Category', 'Risk Score'],
        'Value': [
            'Positive' if diagnosis_pred == 1 else 'Negative',
            f"{diagnosis_prob[0]*100:.1f}%",
            f"{diagnosis_prob[1]*100:.1f}%",
            f"{updrs_pred:.1f}",
            severity,
            f"{risk_score:.1f}/100"
        ]
    })
    
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="parkinsons_prediction.csv",
        mime="text/csv"
    )

else:
    st.info("👈 Please enter patient information in the sidebar and click **Predict Diagnosis** to see results.")
    
    # Show sample patient
    st.subheader("📋 Sample Patient Data")
    st.markdown("""
    This system uses **29 clinical features** to predict:
    - **Diagnosis**: Whether a patient has Parkinson's disease
    - **UPDRS Score**: Severity of symptoms (0-176 scale)
    - **Severity Category**: Mild, Moderate, or Severe
    
    **Features used:**
    - Demographics: Age, Gender, BMI
    - Lifestyle: Smoking, Alcohol, Physical Activity, Diet, Sleep
    - Medical History: Family History, TBI, Hypertension, Diabetes, Depression, Stroke
    - Clinical Measurements: Blood Pressure, Cholesterol levels
    - Cognitive: MoCA Score, Functional Assessment
    - Symptoms: Tremor, Rigidity, Bradykinesia, Postural Instability, Speech Problems, Sleep Disorders, Constipation
    """)

# Footer
st.markdown("---")
st.caption("⚠️ **Disclaimer**: This tool is for research and educational purposes only. Not for clinical diagnosis without professional medical consultation.")
