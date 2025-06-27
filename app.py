import streamlit as st
import pandas as pd
import joblib
import openai

# Load your model and preprocessor
model = joblib.load("diabetes_model_tuned.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Set your OpenAI API key (replace 'YOUR_API_KEY' with your actual key)
openai.api_key = "YOUR_API_KEY"

st.title("Diabetes Prediction System with AI Health Assistant")

# Input fields
glucose = st.number_input("Glucose", min_value=0.0, step=0.1, format="%.2f")
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, step=0.1, format="%.2f")
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, step=0.1, format="%.2f")
insulin = st.number_input("Insulin", min_value=0.0, step=0.1, format="%.2f")
bmi = st.number_input("BMI", min_value=0.0, step=0.1, format="%.2f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.0001, format="%.4f")
age = st.number_input("Age", min_value=0, step=1)
bmi_category = st.number_input("BMI Category (numeric)", min_value=0, max_value=3, step=1, help="0=Underweight, 1=Normal, 2=Overweight, 3=Obese")

# Prediction button
if st.button("Predict Diabetes Risk"):
    input_dict = {
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age,
        'BMI_Category': bmi_category
    }
    input_df = pd.DataFrame([input_dict])
    
    try:
        input_processed = preprocessor.transform(input_df)
        prediction = model.predict(input_processed)[0]
        pred_prob = model.predict_proba(input_processed)[0][1]
        
        if prediction == 1:
            st.error(f"Prediction: Diabetic (Probability: {pred_prob*100:.2f}%)")
            suggestions = [
                "Maintain a balanced diet rich in vegetables and whole grains.",
                "Exercise regularly for at least 30 minutes daily.",
                "Monitor your blood sugar levels frequently.",
                "Avoid sugary foods and beverages.",
                "Consult your healthcare provider regularly.",
                "Stay hydrated and get enough sleep.",
                "Manage stress through meditation or hobbies.",
                "Avoid smoking and limit alcohol consumption.",
                "Take medications as prescribed by your doctor.",
                "Attend regular diabetes education sessions."
            ]
        else:
            st.success(f"Prediction: Non-Diabetic (Probability: {pred_prob*100:.2f}%)")
            suggestions = [
                "Maintain a healthy and balanced diet.",
                "Keep up regular physical activity.",
                "Get regular health checkups.",
                "Manage stress effectively.",
                "Avoid excessive sugar and processed foods.",
                "Maintain a healthy weight.",
                "Limit alcohol consumption.",
                "Avoid smoking.",
                "Stay hydrated.",
                "Get adequate sleep."
            ]
        
        st.markdown("### Suggestions to improve/manage your health:")
        for s in suggestions:
            st.markdown(f"- {s}")
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.markdown("---")

# AI Chat / Advice section
st.subheader("AI Health Assistant")

user_question = st.text_input("Ask your health-related question or get advice:")

if user_question:
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful health assistant."},
                {"role": "user", "content": user_question}
            ],
            max_tokens=300,
            temperature=0.7,
        )
        answer = response.choices[0].message.content.strip()
        st.info(answer)
    except Exception as e:
        st.error(f"AI Response Error: {e}")
