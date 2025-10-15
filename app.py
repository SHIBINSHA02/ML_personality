# app.py

import streamlit as st
from model import PersonalityModel

personality_model = PersonalityModel(data_path='data.csv')
personality_model.train_model()


st.title("Introvert vs Extrovert Predictor")
st.write("Enter the details below to predict personality:")

time_alone = st.number_input("Time Spent Alone (hours/day)", min_value=0, max_value=24, value=5)
stage_fear = st.selectbox("Stage Fear?", ("Yes", "No"))
social_event = st.number_input("Social Event Attendance (per week)", min_value=0, max_value=20, value=2)
going_outside = st.number_input("Going Outside (hours/day)", min_value=0, max_value=24, value=3)
drained_after_socializing = st.selectbox("Feel Drained After Socializing?", ("Yes", "No"))
friends_circle = st.number_input("Friends Circle Size", min_value=0, max_value=50, value=5)
post_frequency = st.number_input("Social Media Post Frequency (per week)", min_value=0, max_value=50, value=3)

if st.button("Predict Personality"):
    input_data = {
        'Time_spent_Alone': time_alone,
        'Stage_fear': stage_fear,
        'Social_event_attendance': social_event,
        'Going_outside': going_outside,
        'Drained_after_socializing': drained_after_socializing,
        'Friends_circle_size': friends_circle,
        'Post_frequency': post_frequency
    }
    result = personality_model.predict(input_data)
    st.success(f"The predicted personality is: **{result}**")

# Optional: Show model coefficients
if st.checkbox("Show Model Coefficients"):
    coeff = personality_model.model.coef_[0]
    features = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 'Going_outside',
                'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']
    coef_dict = dict(zip(features, coeff))
    st.write(coef_dict)
    st.write(f"Intercept: {personality_model.model.intercept_[0]:.4f}")
