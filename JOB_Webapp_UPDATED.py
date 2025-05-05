
import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load skills from dataset
@st.cache_data
def load_skills():
    df = pd.read_csv("Job_Dataset.csv")
    skills = set()
    df['skills'].dropna().apply(lambda x: skills.update(x.split(";")))
    return sorted(skills)

skill_list = load_skills()

# Streamlit app
st.title("Job Offer Prediction for Fresh Graduates")

experience = st.slider("Years of Experience", 0, 10, 1)
grades = st.slider("Average Course Grades", 50, 100, 75)
projects = st.slider("Number of Completed Projects", 0, 20, 2)
extracurriculars = st.slider("Number of Extracurricular Activities", 0, 10, 1)

# Updated skill selection
selected_skill = st.selectbox("Select your skill", skill_list)
skill_index = skill_list.index(selected_skill)

# Prepare input
user_input = [[experience, grades, projects, extracurriculars, skill_index]]
user_input_scaled = scaler.transform(user_input)

# Predict
if st.button("Predict Job Offer"):
    prediction = model.predict(user_input_scaled)
    if prediction[0] == 1:
        st.success("The candidate is likely to receive a job offer.")
    else:
        st.error("The candidate is unlikely to receive a job offer.")
