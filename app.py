import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

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
def load_trained_model():
    """Load pre-trained model from .pkl file"""
    
    model_path = 'pd_classifier_model.pkl'
    
    # Check if model file exists
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found!")
        st.info("Please ensure the trained model file is in the same directory as app.py")
        return None
    
    try:
        clf_model = joblib.load(model_path)
        st.success(f"✅ Model loaded successfully from {model_path}")
        return clf_model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

@st.cache_resource
def get_model_stats():
    """Calculate model stats from test set"""
    df = pd.read_csv('parkinsons_disease_data.csv')
    X = df[FEATURE_COLUMNS]
    y_diagnosis = df['Diagnosis']
    
    # Load model and calculate accuracy
    clf_model = joblib.load('pd_classifier_model.pkl')
    accuracy = clf_model.score(X, y_diagnosis)
    
    return accuracy

# Load the trained model
clf_model = load_trained_model()

if clf_model is None:
    st.stop()

# Get model accuracy
try:
    clf_accuracy = get_model_stats()
except:
    clf_accuracy = 0.0

# Title
st.title("🧠 Parkinson's Disease Detection & Severity Prediction")
st.markdown("**ML-Based Clinical Decision Support System (Using Pre-trained Model)**")

# Model performance
st.metric("Classification Model Accuracy", f"{clf_accuracy:.2%}")

st.markdown("---")

# Sidebar inputs
st.sidebar.header("📋 Patient Information")

# Demographics - CORRECTED DEFAULTS (Healthy)
st.sidebar.subheader("Demographics")
age = st.sidebar.slider("Age (years)", 18, 100, 65)
gender = st.sidebar.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
bmi = st.sidebar.slider("BMI", 10.0, 60.0, 24.0, 0.1)

# Lifestyle - HEALTHY DEFAULTS
st.sidebar.subheader("Lifestyle Factors")
smoking = st.sidebar.selectbox("Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
alcohol = st.sidebar.slider("Alcohol Consumption", 0.0, 10.0, 2.0, 0.1)
physical_activity = st.sidebar.slider("Physical Activity", 0.0, 10.0, 7.0, 0.1)
diet_quality = st.sidebar.slider("Diet Quality", 0.0, 10.0, 7.0, 0.1)
sleep_quality = st.sidebar.slider("Sleep Quality", 0.0, 10.0, 7.0, 0.1)

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
cholesterol_total = st.sidebar.slider("Total Cholesterol", 100, 400, 180)
cholesterol_ldl = st.sidebar.slider("LDL Cholesterol", 50, 250, 100)
cholesterol_hdl = st.sidebar.slider("HDL Cholesterol", 20, 100, 55)
cholesterol_triglycerides = st.sidebar.slider("Triglycerides", 50, 400, 100)

# Cognitive & Functional
st.sidebar.subheader("Cognitive & Functional")
moca = st.sidebar.slider("MoCA Score", 0, 30, 26)
functional_assessment = st.sidebar.slider("Functional Assessment", 0.0, 10.0, 8.0, 0.1)

# Symptoms - HEALTHY DEFAULTS (No symptoms)
st.sidebar.subheader("Symptoms")
tremor = st.sidebar.slider("Tremor (0-10)", 0, 10, 0)
rigidity = st.sidebar.slider("Rigidity (0-10)", 0, 10, 0)
bradykinesia = st.sidebar.slider("Bradykinesia (0-10)", 0, 10, 0)
postural_instability = st.sidebar.slider("Postural Instability (0-10)", 0, 10, 0)
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
    
    try:
        # Make predictions
        diagnosis_pred = clf_model.predict(input_data)[0]
        diagnosis_prob = clf_model.predict_proba(input_data)[0]
        
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
        
        # Feature importance
        st.subheader("🔍 Feature Importance (Top 10)")
        try:
            feature_importance = clf_model.get_feature_importance()
            importance_df = pd.DataFrame({
                'Feature': FEATURE_COLUMNS,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False).head(10)
            
            st.bar_chart(importance_df.set_index('Feature'))
        except:
            st.info("Feature importance visualization not available for this model type.")
        
        st.markdown("---")
        
        # Download results
        results_df = pd.DataFrame({
            'Metric': ['Diagnosis', 'Confidence (Negative)', 'Confidence (Positive)'],
            'Value': [
                'Positive' if diagnosis_pred == 1 else 'Negative',
                f"{diagnosis_prob[0]*100:.1f}%",
                f"{diagnosis_prob[1]*100:.1f}%"
            ]
        })
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="parkinsons_prediction.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")

else:
    st.info("👈 Please enter patient information in the sidebar and click **Predict Diagnosis** to see results.")
    
    # Show sample patient
    st.subheader("📋 About This System")
    st.markdown("""
    This system uses a **pre-trained machine learning model** to predict Parkinson's disease based on **29 clinical features**.
    
    **Features used:**
    - **Demographics**: Age, Gender, BMI
    - **Lifestyle**: Smoking, Alcohol, Physical Activity, Diet, Sleep
    - **Medical History**: Family History, TBI, Hypertension, Diabetes, Depression, Stroke
    - **Clinical Measurements**: Blood Pressure, Cholesterol levels
    - **Cognitive**: MoCA Score, Functional Assessment
    - **Symptoms**: Tremor, Rigidity, Bradykinesia, Postural Instability, Speech Problems, Sleep Disorders, Constipation
    
    **Default values** are set to represent a healthy individual. Adjust the sliders to match your patient's profile.
    """)

# Footer
st.markdown("---")
st.caption("⚠️ **Disclaimer**: This tool is for research and educational purposes only. Not for clinical diagnosis without professional medical consultation.")
st.caption("Model used: pd_classifier_model.pkl")
