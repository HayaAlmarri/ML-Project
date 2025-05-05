import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open('best_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

def main():
    st.set_page_config(page_title="Job Offer Predictor", layout="centered")

    st.title("🎯 Job Offer Prediction App")
    st.markdown("This tool predicts whether a graduate is likely to receive a job offer based on profile attributes.")

    # User Inputs
    experience_years = st.slider("Years of Experience", 0, 10, 0)
    course_grades = st.slider("Course Grade (0–100)", 0, 100, 70)
    projects_completed = st.slider("Number of Completed Projects", 0, 20, 3)
    extracurriculars = st.slider("Extracurricular Activities (0–10)", 0, 10, 2)
    skills_encoded = st.selectbox("Skill Category", [0, 1, 2, 3, 4, 5, 6], index=0)

    # Prepare input
    input_data = np.array([[experience_years, course_grades, projects_completed, extracurriculars, skills_encoded]])
    feature_names = ['experience_years', 'course_grades', 'projects_completed', 'extracurriculars', 'skills_encoded']
    input_df = pd.DataFrame(input_data, columns=feature_names)
    input_scaled = scaler.transform(input_df)

    if st.button("🔍 Predict Job Offer"):
        prediction = model.predict(input_scaled)[0]
        if prediction == 1:
            st.success("✅ The candidate is **likely to receive a job offer**.")
        else:
            st.error("❌ The candidate is **unlikely to receive a job offer**.")

if __name__ == '__main__':
    main()
