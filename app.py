# app.py
import streamlit as st
from model import PersonalityModel
import numpy as np
import matplotlib.pyplot as plt

# Train the model
personality_model = PersonalityModel(data_path='data.csv')
personality_model.train_model()

st.title("🧠 Introvert vs Extrovert Personality Predictor")
st.write("Enter your details below to predict whether you're more **Introverted** or **Extroverted** based on behavioral traits.")

# Input fields
time_alone = st.number_input("⏱️ Time Spent Alone (hours/day)", min_value=0, max_value=24, value=5)
stage_fear = st.selectbox("🎤 Stage Fear?", ("Yes", "No"))
social_event = st.number_input("🎉 Social Event Attendance (per week)", min_value=0, max_value=20, value=2)
going_outside = st.number_input("🚶 Going Outside (hours/day)", min_value=0, max_value=24, value=3)
drained_after_socializing = st.selectbox("😩 Feel Drained After Socializing?", ("Yes", "No"))
friends_circle = st.number_input("👥 Friends Circle Size", min_value=0, max_value=50, value=5)
post_frequency = st.number_input("📱 Social Media Post Frequency (per week)", min_value=0, max_value=50, value=3)

# Prediction
if st.button("🔍 Predict Personality"):
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

# Show coefficients
if st.checkbox("📊 Show Model Coefficients"):
    coef_dict = personality_model.get_coefficients()
    st.subheader("Model Coefficients (Feature Importance)")
    st.table(coef_dict)

    # Intercept
    logreg_step = personality_model.model.named_steps['logreg']
    st.write(f"**Intercept:** {logreg_step.intercept_[0]:.4f}")

# Show mathematical representation
if st.checkbox("🧮 Show Mathematical Representation"):
    logreg_step = personality_model.model.named_steps['logreg']
    intercept = logreg_step.intercept_[0]
    coef = logreg_step.coef_[0]
    feature_names = ['Time_spent_Alone','Stage_fear','Social_event_attendance',
                     'Going_outside','Drained_after_socializing','Friends_circle_size',
                     'Post_frequency']

    # Build equation dynamically
    equation = f"P(Personality=1) = 1 / (1 + exp(-({intercept:.3f}"
    for fname, c in zip(feature_names, coef):
        equation += f" + ({c:.3f}×{fname})"
    equation += ")))"

    st.latex(equation)

# Plot data and model probabilities
if st.checkbox("📈 Show Feature-wise Probability Graph"):
    df = personality_model.load_and_preprocess_data()

    selected_feature = st.selectbox(
        "Select a feature to visualize:",
        ['Time_spent_Alone','Stage_fear','Social_event_attendance',
         'Going_outside','Drained_after_socializing','Friends_circle_size',
         'Post_frequency']
    )

    X = df[['Time_spent_Alone','Stage_fear','Social_event_attendance',
            'Going_outside','Drained_after_socializing','Friends_circle_size',
            'Post_frequency']]
    y = df['Personality']

    feature_idx = X.columns.get_loc(selected_feature)
    x_values = np.linspace(X[selected_feature].min(), X[selected_feature].max(), 100)

    # Sweep feature across range while others are mean
    X_mean = X.mean().values
    X_sweep = np.tile(X_mean, (100, 1))
    X_sweep[:, feature_idx] = x_values

    probs = personality_model.model.predict_proba(X_sweep)[:, 1]

    fig, ax = plt.subplots()
    ax.scatter(X[selected_feature], y, alpha=0.4, label="Actual Personality")
    ax.plot(x_values, probs, color='red', linewidth=2, label='Predicted Probability')
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("P(Personality=1)")
    ax.set_title(f"Logistic Regression Fit for {selected_feature}")
    ax.legend()
    st.pyplot(fig)
