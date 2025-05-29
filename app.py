import streamlit as st
import joblib
import json
import pandas as pd

# Training data age mean and standard deviation
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

# App title
st.title("COPD Patient Cardiovascular Disease (CVD) Risk Prediction")

# Introductory section
intro_col1, intro_col2 = st.columns([1, 1])
with intro_col1:
    st.image("https://img.icons8.com/fluency/96/000000/lungs.png", width=120)
    st.subheader("What is COPD?")
    st.write(
        "**Chronic Obstructive Pulmonary Disease (COPD)** is a progressive lung disease characterized by airflow limitation, "
        "chronic inflammation, and breathing difficulties. Common causes include long-term exposure to cigarette smoke, environmental pollutants, and occupational dust."
    )
    st.write("Symptoms: chronic cough, sputum production, dyspnea.")
with intro_col2:
    st.image("https://img.icons8.com/fluency/96/000000/heart-with-pulse.png", width=120)
    st.subheader("What is CVD?")
    st.write(
        "**Cardiovascular Disease (CVD)** encompasses disorders of the heart and blood vessels, including coronary artery disease, heart failure, and stroke. "
        "Major risk factors: hypertension, dyslipidemia, diabetes, smoking."
    )
    st.write("Impact: leading cause of death globally.")

# First divider
st.markdown("---")

# Clinical significance
st.markdown("## Importance of Predicting CVD Risk in COPD Patients")
st.write(
    "COPD and CVD often coexist; systemic inflammation and shared risk factors accelerate cardiovascular complications in COPD. "
    "Early identification of high-risk COPD patients for CVD can guide preventive measures, optimize management, and improve outcomes."
)
# Info banner
st.info(
    "⚡ Enter the key features on the left and click **Predict Risk** to view the risk assessment below!"
)

st.info(
    "⚡  Slide down to view the results ⬇️⬇️⬇️"
)

# Arrow prompt immediately after info if submitted
def show_arrow():
    st.markdown(
        "<div style='font-size:48px; text-align:center;'>⬇️</div>"
        "<div style='text-align:center; font-weight:bold;'>Slide down to view the results</div>",
        unsafe_allow_html=True
    )

# Second divider
st.markdown("---")

# Sidebar – Input Features
st.sidebar.header("Input Features")
with st.sidebar.form("predict_form"):
    iadl_options = [f"{i} items with difficulties" for i in range(6)]
    iadl_map     = {opt: i for i, opt in enumerate(iadl_options)}
    user_iadl    = st.selectbox("IADL Score", iadl_options)

    gender_options = ["Female", "Male"]
    gender_map     = {"Female": 0, "Male": 1}
    user_gender    = st.selectbox("Gender", gender_options)

    yn_features = [
        "Residence", "Hypertension", "Dyslipidemia", "Digestive disease",
        "Vigorous activity", "Moderate activity", "Disability status", "Tap water access"
    ]
    yn_map  = {"No": 0, "Yes": 1}
    user_yn = {feat: st.selectbox(feat, ["No", "Yes"]) for feat in yn_features}

    health_options = ["Very poor", "Poor", "Average", "Good", "Very Good"]
    health_map     = {opt: i+1 for i, opt in enumerate(health_options)}
    user_health    = st.selectbox("Self-rated Health", health_options)
    user_hearing   = st.selectbox("Hearing", health_options)

    user_age  = st.number_input("Age", min_value=0.0, step=1.0, value=age_mean)
    submitted = st.form_submit_button("Predict Risk")

# Show arrow under info banner if form submitted
if 'submitted' in locals() and submitted:
    show_arrow()

# Prediction and results
if submitted:
    # Prepare features
    data = {
        "IADL score": iadl_map[user_iadl],
        "Gender":     gender_map[user_gender],
    }
    data.update({feat: yn_map[user_yn[feat]] for feat in yn_features})
    data["Self rated health"] = health_map[user_health]
    data["Hearing"]           = health_map[user_hearing]
    data["Age"]               = z_score(user_age, age_mean, age_std)

    # Build DataFrame & predict
    df        = pd.DataFrame([[data[f] for f in features]], columns=features)
    risk_prob = model.predict_proba(df)[0, 1]
    risk_pct  = risk_prob * 100

    # Anchor just above results
    st.markdown("<div id='results_anchor'></div>", unsafe_allow_html=True)

    # Display Risk Assessment
    st.subheader("📊 Risk Assessment")
    st.metric(label="CVD Risk Probability", value=f"{risk_pct:.2f}%")
    st.progress(risk_prob)

    # Tiered Recommendations
    st.subheader("🔍 Tiered Recommendations")
    if risk_prob < 0.3:
        st.success("🟢 Low Risk: Maintain current healthy habits and regular monitoring.")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image("https://img.icons8.com/color/96/000000/running.png", width=80)
            st.markdown("**Exercise Maintenance**")
            st.write("- 30 minutes of moderate exercise daily", "- Activities like walking or tai chi.")
        with c2:
            st.image("https://img.icons8.com/color/96/000000/vegetarian-food.png", width=80)
            st.markdown("**Balanced Nutrition**")
            st.write("- High fiber, low salt and fat diet", "- Include vegetables, whole grains, and lean protein.")
        with c3:
            st.image("https://img.icons8.com/color/96/000000/heart-monitor.png", width=80)
            st.markdown("**Routine Monitoring**")
            st.write("- Monthly blood pressure and heart rate checks", "- Log and observe any anomalies.")
    elif risk_prob <= 0.7:
        st.info("🟡 Moderate Risk: Enhance self-management and consult healthcare providers regularly.")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image("https://img.icons8.com/color/96/000000/yoga.png", width=80)
            st.markdown("**Enhanced Exercise**")
            st.write("- 40 minutes of moderate to vigorous aerobic exercise daily", "- Incorporate resistance training like bands.")
        with c2:
            st.image("https://img.icons8.com/color/96/000000/meal.png", width=80)
            st.markdown("**Nutritional Adjustment**")
            st.write("- Limit processed foods and sugars", "- Increase omega-3 rich foods.")
        with c3:
            st.image("https://img.icons8.com/color/96/000000/doctor-male.png", width=80)
            st.markdown("**Regular Follow-up**")
            st.write("- Quarterly checks: blood pressure, lipid panel, ECG", "- Discuss possible medication adjustments.")
    else:
        st.warning("🔴 High Risk: Seek immediate medical evaluation for specialized cardiovascular assessment.")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image("https://img.icons8.com/color/96/000000/stethoscope.png", width=80)
            st.markdown("**Specialized Testing**")
            st.write("- Comprehensive cardiovascular tests: echocardiography, coronary CT", "- Vascular function and inflammation marker assessment.")
        with c2:
            st.image("https://img.icons8.com/color/96/000000/pill.png", width=80)
            st.markdown("**Medication Management**")
            st.write("- Adhere to antihypertensive and statin therapy", "- Monitor for side effects and efficacy.")
        with c3:
            st.image("https://img.icons8.com/color/96/000000/no-smoking.png", width=80)
            st.markdown("**Lifestyle Intervention**")
            st.write("- Cease smoking and avoid secondhand smoke", "- Maintain regular sleep schedule and stress management.")

    # Smooth scroll to the anchor after a short delay
    st.markdown(
        """
        <script>
        setTimeout(function(){
            var anchor = document.getElementById('results_anchor');
            if(anchor){
                anchor.scrollIntoView({behavior: 'smooth'});
            }
        }, 100);
        </script>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# 用普通的 markdown 输出，不带背景框
st.markdown(
    "*These predictions are for informational purposes only. "
    "Consult healthcare professionals for tailored advice.*"
)
