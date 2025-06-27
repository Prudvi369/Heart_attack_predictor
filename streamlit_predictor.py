import streamlit as st
import pickle
import pandas as pd

# Load model
model_path = "heart_disease_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")
st.title("🫀 Heart Disease Risk Predictor")
st.markdown("Enter patient health details to estimate the risk of heart disease.")

# --- Optional Debug Mode ---
debug_mode = st.checkbox("🧪 Enable Debug Mode")

# --- Inputs ---
st.subheader("📋 Patient Information")

age = st.slider("Age", 18, 85, 50)
sex = st.selectbox("Sex", ["Female", "Male"])
cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
trestbps = st.slider("Resting BP (mm Hg)", 90, 200, 120)
chol = st.slider("Cholesterol (mg/dL)", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
restecg = st.selectbox("Resting ECG", ["Normal", "ST-T abnormality", "Left ventricular hypertrophy"])
thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)
exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.2, 1.0, step=0.1)
slope = st.selectbox("Slope of ST segment", ["Upsloping", "Flat", "Downsloping"])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", ["Normal", "Fixed defect", "Reversible defect"])

# --- Input to model ---
input_dict = {
    "age": age,
    "sex": 1 if sex == "Male" else 0,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": 1 if fbs == "Yes" else 0,
    "thalach": thalach,
    "exang": 1 if exang == "Yes" else 0,
    "oldpeak": oldpeak,
    "ca": ca,
    "restecg_1": 1 if restecg == "ST-T abnormality" else 0,
    "restecg_2": 1 if restecg == "Left ventricular hypertrophy" else 0,
    "slope_1": 1 if slope == "Flat" else 0,
    "slope_2": 1 if slope == "Downsloping" else 0,
    "thal_1": 1 if thal == "Fixed defect" else 0,
    "thal_2": 1 if thal == "Reversible defect" else 0,
    "thal_3": 0,
    "cp_1": 1 if cp == "Atypical Angina" else 0,
    "cp_2": 1 if cp == "Non-Anginal Pain" else 0,
    "cp_3": 1 if cp == "Asymptomatic" else 0,
}

input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# --- Prediction ---
if st.button("🔍 Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    # ✅ Show Confidence Meter
    st.subheader("📊 Prediction Confidence")
    st.progress(min(int(prob * 100), 100), text=f"Confidence: {prob:.2%}")

    # ✅ Risk Result
    if pred == 1:
        st.error(f"⚠️ High Risk of Heart Disease\n💡 Confidence: {prob:.2%}")
    else:
        st.success(f"✅ Low Risk of Heart Disease\n💡 Confidence: {prob:.2%}")

    # ✅ Optional Debug Table
    if debug_mode:
        st.markdown("### 🧪 Debug Input to Model")
        st.dataframe(input_df.T.style.applymap(
            lambda v: "color: red;" if v in [1, True] else "color: green;"
        ))

        st.markdown(
            """
            <small><i>Red = Risky feature present</i></small>
            """,
            unsafe_allow_html=True
        )
