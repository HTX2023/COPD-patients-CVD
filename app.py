import streamlit as st
import joblib
import json
import pandas as pd

# ----------------------------
# Training data Age mean and standard deviation
age_mean = 65.2599
age_std  = 9.0775

def z_score(x, mean, std):
    return (x - mean) / std

# Load model and feature names
model = joblib.load('SVM.pkl')
with open('feature_names.json', 'r') as f:
    features = json.load(f)

# Page configuration
st.set_page_config(page_title="COPD-CVD Risk Prediction", layout="wide")
st.title("COPD Patient Cardiovascular Disease (CVD) Risk Prediction")
st.markdown(
    "This tool uses a pre-trained SVM model to predict the probability of CVD in COPD patients "
    "based on clinical features, and provides tiered, visualized health management advice."
)

# Sidebar - Input Features
st.sidebar.header("Input Features")
with st.sidebar.form("predict_form"):
    # IADL score: 0–5
    iadl_opts = [f"{i} items with difficulties" for i in range(6)]
    map_iadl  = {opt: i for i, opt in enumerate(iadl_opts)}
    user_iadl = st.selectbox("IADL Score", iadl_opts)

    # Gender: 0=Female, 1=Male
    gender_opts = ["Female", "Male"]
    map_gender  = {"Female": 0, "Male": 1}
    user_gender = st.selectbox("Gender", gender_opts)

    # Yes/No features: 0=No, 1=Yes
    yn_feats = [
        "Residence", "Hypertension", "Dyslipidemia", "Digestive disease",
        "Vigorous activity", "Moderate activity", "Disability status", "Tap water access"
    ]
    map_yn = {"No": 0, "Yes": 1}
    user_yn = {feat: st.selectbox(feat, ["No", "Yes"]) for feat in yn_feats}

    # Self-rated health & Hearing: Very poor–Very Good
    health_opts = ["Very poor", "Poor", "Average", "Good", "Very Good"]
    map_health  = {opt: i+1 for i, opt in enumerate(health_opts)}
    user_health  = st.selectbox("Self-rated Health", health_opts)
    user_hearing = st.selectbox("Hearing", health_opts)

    # Age (continuous input)
    user_age = st.number_input("Age", min_value=0.0, step=1.0, value=age_mean)

    submitted = st.form_submit_button("Predict Risk")

# Prediction and Tiered Advice
if submitted:
    # Prepare input vector
    data = {
        "IADL score": map_iadl[user_iadl],
        "Gender":     map_gender[user_gender],
    }
    data.update({feat: map_yn[user_yn[feat]] for feat in yn_feats})
    data["Self rated health"] = map_health[user_health]
    data["Hearing"]           = map_health[user_hearing]
    data["Age"]               = z_score(user_age, age_mean, age_std)

    # Create DataFrame for prediction
    df = pd.DataFrame([[data[f] for f in features]], columns=features)
    risk_prob = model.predict_proba(df)[0, 1]
    risk_pct  = risk_prob * 100

    # Display risk gauge
    st.subheader("📊 Risk Assessment")
    st.metric(label="CVD Risk Probability", value=f"{risk_pct:.2f}%")
    st.progress(risk_prob)

    # Tiered advice: <0.3 Low, 0.3-0.7 Moderate, >0.7 High
    st.subheader("🔍 Tiered Recommendations")
    if risk_prob < 0.3:
        # Low risk
        st.success("🟢 Low Risk: Maintain current healthy habits and regular monitoring.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("https://img.icons8.com/color/96/000000/running.png", width=80)
            st.markdown("**Exercise Maintenance**")
            st.write("- 30 minutes of moderate exercise daily",
                     "- Activities like walking or tai chi.")
        with col2:
            st.image("https://img.icons8.com/color/96/000000/vegetarian-food.png", width=80)
            st.markdown("**Balanced Nutrition**")
            st.write("- High fiber, low salt and fat diet",
                     "- Include vegetables, whole grains, and lean protein.")
        with col3:
            st.image("https://img.icons8.com/color/96/000000/heart-monitor.png", width=80)
            st.markdown("**Routine Monitoring**")
            st.write("- Monthly blood pressure and heart rate checks",
                     "- Log and observe any anomalies.")
    elif risk_prob <= 0.7:
        # Moderate risk
        st.info("🟡 Moderate Risk: Enhance self-management and consult healthcare providers regularly.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("https://img.icons8.com/color/96/000000/yoga.png", width=80)
            st.markdown("**Enhanced Exercise**")
            st.write("- 40 minutes of moderate to vigorous aerobic exercise daily",
                     "- Incorporate resistance training like bands.")
        with col2:
            st.image("https://img.icons8.com/color/96/000000/meal.png", width=80)
            st.markdown("**Nutritional Adjustment**")
            st.write("- Limit processed foods and sugars",
                     "- Increase omega-3 rich foods.")
        with col3:
            st.image("https://img.icons8.com/color/96/000000/doctor-male.png", width=80)
            st.markdown("**Regular Follow-up**")
            st.write("- Quarterly checks: blood pressure, lipid panel, ECG",
                     "- Discuss possible medication adjustments.")
    else:
        # High risk
        st.warning("🔴 High Risk: Seek immediate medical evaluation for specialized cardiovascular assessment.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("https://img.icons8.com/color/96/000000/stethoscope.png", width=80)
            st.markdown("**Specialized Testing**")
            st.write("- Comprehensive cardiovascular tests: echocardiography, coronary CT",
                     "- Vascular function and inflammation marker assessment.")
        with col2:
            st.image("https://img.icons8.com/color/96/000000/pill.png", width=80)
            st.markdown("**Medication Management**")
            st.write("- Adhere to antihypertensive and statin therapy",
                     "- Monitor for side effects and efficacy.")
        with col3:
            st.image("https://img.icons8.com/color/96/000000/no-smoking.png", width=80)
            st.markdown("**Lifestyle Intervention**")
            st.write("- Cease smoking and avoid secondhand smoke",
                     "- Maintain regular sleep schedule and stress management.")

    st.markdown("---")
    st.write("*These recommendations are for informational purposes only. Consult healthcare professionals for tailored advice.*")
