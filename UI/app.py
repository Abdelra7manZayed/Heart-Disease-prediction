import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and assets
@st.cache_resource
def load_assets():
    model = joblib.load('models/final_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    with open('models/selected_features.txt', 'r') as f:
        features = [line.strip() for line in f]
    return model, scaler, features

model, scaler, selected_features = load_assets()

# Prediction logic
def predict_heart_disease(model, scaler, features, input_data):
    input_df = pd.DataFrame({feature: [0] for feature in features})

    for key in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak',
                'sex', 'fbs', 'exang', 'ca', 'restecg']:
        input_df[key] = input_data[key]

    # One-hot encodings
    cp_map = {0: 'cp_1', 1: 'cp_2', 2: 'cp_3', 3: 'cp_4'}
    slope_map = {0: 'slope_1', 1: 'slope_2', 2: 'slope_3'}
    thal_map = {0: 'thal_6.0', 1: 'thal_7.0', 2: 'thal_3'}

    if input_data['cp'] in cp_map:
        input_df[cp_map[input_data['cp']]] = 1
    if input_data['slope'] in slope_map:
        input_df[slope_map[input_data['slope']]] = 1
    if input_data['thal'] in thal_map:
        input_df[thal_map[input_data['thal']]] = 1

    # Scale numerical features
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    input_df = input_df[features]

    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    return prediction, prediction_proba

# Streamlit UI
st.set_page_config(page_title="Heart Health Assessment", layout="centered", page_icon="❤️")
st.title("❤️ Heart Health Assessment Tool")
st.markdown("""
This simple assessment estimates your heart disease risk based on key health indicators.  
**Please note:** This is not a medical diagnosis. Always consult with your healthcare provider.
""")

# --- Form ---
with st.form("health_form"):
    st.header("Your Health Information")
    st.write("Please provide the following details to assess your heart health:")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Your age", 20, 100, 50, 
                       help="Age is an important factor in heart health assessment")
        
        sex = st.radio("Gender", ["Male", "Female", "Prefer not to say"],
                      help="Biological sex can influence heart disease risk factors")
        
        trestbps = st.slider("Resting blood pressure (mm Hg)", 80, 200, 120,
                            help="Your blood pressure when at rest")
        
        chol = st.slider("Cholesterol level (mg/dL)", 100, 600, 200,
                         help="Total cholesterol level from your most recent blood test")
        
        thalach = st.slider("Highest heart rate achieved during exercise", 70, 210, 150,
                           help="If you don't know, estimate based on your exercise experience")
        
        oldpeak = st.slider("ST depression (if known)", 0.0, 6.0, 1.0, step=0.1,
                           help="From electrocardiogram (ECG) results, if available")

    with col2:
        fbs = st.radio("Fasting blood sugar > 120 mg/dl?", ["No", "Yes", "Not sure"],
                      help="From your most recent blood test")
        
        exang = st.radio("Do you experience chest pain during exercise?", ["No", "Yes", "Sometimes"],
                         help="Known as 'exercise-induced angina'")
        
        st.subheader("If you have ECG results:")
        restecg_options = {
            "Normal": 0,
            "ST-T wave abnormality": 1,
            "Possible left ventricular hypertrophy": 2
        }
        restecg = st.selectbox("Resting ECG results", list(restecg_options.keys()),
                               help="From your most recent electrocardiogram")
        
        cp_options = {
            "No chest pain": 0,
            "Typical angina (chest pain related to heart)": 1,
            "Atypical angina (other chest pain)": 2,
            "Non-anginal pain (not heart-related)": 3
        }
        cp = st.selectbox("Chest pain type (if any)", list(cp_options.keys()))
        
        slope_options = {
            "Upsloping": 0,
            "Flat": 1,
            "Downsloping": 2
        }
        slope = st.selectbox("ST segment slope (from ECG)", list(slope_options.keys()),
                            help="From electrocardiogram results")
        
        thal_options = {
            "Normal blood flow": 0,
            "Fixed defect (reduced blood flow)": 1,
            "Reversible defect (variable blood flow)": 2
        }
        thal = st.selectbox("Thalassemia (if known)", list(thal_options.keys()),
                          help="A blood disorder that can affect heart health")
        
        ca = st.slider("Number of major vessels seen on fluoroscopy", 0, 3, 0,
                      help="From coronary angiography if performed (0 = none)")

    submitted = st.form_submit_button("Assess My Heart Health")

# --- Prediction ---
if submitted:
    # Convert inputs to model format
    input_data = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "trestbps": trestbps,
        "chol": chol,
        "thalach": thalach,
        "oldpeak": oldpeak,
        "fbs": 1 if fbs == "Yes" else 0,
        "exang": 1 if exang == "Yes" else (0.5 if exang == "Sometimes" else 0),
        "restecg": restecg_options[restecg],
        "cp": cp_options[cp],
        "slope": slope_options[slope],
        "thal": thal_options[thal],
        "ca": ca
    }

    with st.spinner("Analyzing your health information..."):
        try:
            pred, proba = predict_heart_disease(model, scaler, selected_features, input_data)
        except:
            st.error("We encountered an issue processing your information. Please try again or consult your doctor.")
            st.stop()

    # --- Display Results ---
    st.header("Your Heart Health Assessment")
    st.markdown("---")
    
    # Main result card
    with st.container():
        if pred == 1:
            st.error("""
            ## 🔍 Potential Heart Health Concern Detected
            Based on the information provided, this assessment suggests you may be at risk for heart disease.
            """)
        else:
            st.success("""
            ## 🌟 No Significant Heart Health Concerns Detected
            Based on the information provided, this assessment doesn't indicate significant heart disease risk.
            """)
        
        st.info("""
        **Remember:** This is not a medical diagnosis. It's an automated assessment based on the information you provided. 
        Always consult with your healthcare provider about your heart health.
        """)
    
    # Risk visualization
    st.subheader("Risk Assessment Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Healthy Heart Likelihood", value=f"{proba[0]*100:.1f}%")
        st.progress(proba[0])
        
    with col2:
        st.metric(label="Potential Heart Disease Risk", value=f"{proba[1]*100:.1f}%")
        st.progress(proba[1])
    
    # Personalized guidance
    st.subheader("Recommended Next Steps")
    
    if pred == 1:
        if proba[1] > 0.9:
            st.error("""
            **🛑 High Risk Indicated**  
            Please schedule an appointment with your doctor or cardiologist as soon as possible 
            to discuss these results and your heart health.
            """)
        elif proba[1] > 0.7:
            st.warning("""
            **⚠️ Moderate to High Risk Indicated**  
            We recommend consulting with your healthcare provider about these results 
            and considering a heart health check-up.
            """)
        else:
            st.info("""
            **🔍 Slight Risk Indicated**  
            While not immediately concerning, it may be worth discussing these results 
            at your next regular check-up.
            """)
    else:
        if proba[0] > 0.9:
            st.success("""
            **🌟 Excellent Heart Health Indicators**  
            Your results suggest good heart health. Keep up any healthy habits you have!
            """)
        else:
            st.info("""
            **👍 Generally Positive Results**  
            Your results don't indicate significant concerns, but regular check-ups 
            are always a good idea for maintaining heart health.
            """)
    
    # Health tips
    st.subheader("Heart Health Tips for Everyone")
    
    tips = """
    - 🏃‍♂️ Get regular physical activity (150 minutes moderate exercise per week)
    - 🥗 Eat a heart-healthy diet (plenty of vegetables, whole grains, lean proteins)
    - 🚭 Avoid tobacco products
    - 🍷 Limit alcohol consumption
    - 😴 Get 7-9 hours of quality sleep each night
    - 🧘‍♀️ Manage stress through relaxation techniques
    - 🩺 Schedule regular check-ups with your doctor
    """
    st.markdown(tips)
    
    # Data summary (simplified)
    expander = st.expander("Review the information you provided")
    with expander:
        st.json({
            "Demographics": {
                "Age": age,
                "Gender": sex
            },
            "Vital Signs": {
                "Blood Pressure": f"{trestbps} mm Hg",
                "Cholesterol": f"{chol} mg/dL",
                "Max Heart Rate": thalach
            },
            "Symptoms": {
                "Chest Pain": cp,
                "Exercise Pain": exang
            },
            "Test Results": {
                "ECG Findings": restecg,
                "ST Depression": oldpeak,
                "Fluoroscopy Vessels": ca
            }
        })

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer:** This tool is for informational purposes only and is not a substitute for professional medical advice, 
diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any 
questions you may have regarding a medical condition.
""")
