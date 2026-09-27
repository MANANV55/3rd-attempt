import streamlit as st
import joblib
import numpy as np

# --- Page config (pehla Streamlit call) ---
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# --- Custom CSS: theme change ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }
    h1 {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(90deg, #00c9ff, #92fe9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 2.5rem !important;
    }
    h2, h3, p, label, div, span { color: #f0f4f8 !important; }
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #00c9ff !important;
        box-shadow: 0 0 12px #00c9ff !important;
    }
    .stSlider > div > div > div > div > div {
        background: linear-gradient(90deg, #00c9ff, #92fe9d) !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #00c9ff, #92fe9d);
        color: #0f2027 !important;
        border: none;
        border-radius: 12px;
        padding: 0.7em 2.5em;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 25px rgba(0, 201, 255, 0.5);
    }
    [data-testid="stMetricValue"] {
        color: #00c9ff !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] { color: #92fe9d !important; }
    div[data-testid="stAlert"] { border-radius: 12px !important; }
    .css-1d391kg, section[data-testid="stSidebar"] { background: #0f2027; }
</style>
""", unsafe_allow_html=True)

# --- Load models ---
reg_model  = joblib.load("diabetes_reg_model.joblib")
clf_model  = joblib.load("diabetes_clf_model.joblib")
scaler     = joblib.load("diabetes_scaler.joblib")
median_val = joblib.load("diabetes_median.joblib")

# --- Reverse-scaling stats (original diabetes dataset) ---
STATS = {
    "age":               (48.5, 13.1),
    "bmi":               (26.4, 4.4),
    "blood_pressure":    (94.6, 13.8),
    "total_cholesterol": (189.1, 34.8),
    "ldl":               (115.4, 30.4),
    "hdl":               (49.8, 12.9),
    "chol_hdl_ratio":    (4.07, 1.29),
    "triglycerides":     (4.48, 1.28),
    "blood_sugar":       (91.5, 11.5),
}
SEX_SCALED = {"Female": -0.0446, "Male": 0.0507}

def scale(name, value):
    m, s = STATS[name]
    return (value - m) / s

# --- Header ---
st.title("🩺 Diabetes Progression & Risk Predictor")
st.write("Adjust the sliders below. The model will estimate the patient's diabetes progression score and classify their risk level.")
st.caption("⚠️ Educational demo only — not a clinical tool.")

# --- Inputs (realistic ranges) ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Basic Info")
    age    = st.slider("Age (years)",             19, 80, 48)
    sex    = st.selectbox("Sex",                  ["Female", "Male"])
    bmi    = st.slider("BMI (kg/m²)",             18.0, 40.0, 26.4, 0.1)
    bp     = st.slider("Blood Pressure (mm Hg)",  60, 130, 95)
    chol   = st.slider("Total Cholesterol (mg/dL)", 100, 300, 189)

with col2:
    st.markdown("### 🧪 Lab Values")
    ldl    = st.slider("LDL (mg/dL)",             50, 200, 115)
    hdl    = st.slider("HDL (mg/dL)",             20, 100, 50)
    ratio  = st.slider("Chol/HDL Ratio",          2.0, 8.0, 4.1, 0.1)
    trig   = st.slider("Triglycerides (log)",     3.0, 6.0, 4.5, 0.1)
    sugar  = st.slider("Blood Sugar (mg/dL)",     60, 130, 91)

# --- Predict ---
if st.button("🔮 Predict"):
    scaled_input = np.array([[
        scale("age", age),
        SEX_SCALED[sex],
        scale("bmi", bmi),
        scale("blood_pressure", bp),
        scale("total_cholesterol", chol),
        scale("ldl", ldl),
        scale("hdl", hdl),
        scale("chol_hdl_ratio", ratio),
        scale("triglycerides", trig),
        scale("blood_sugar", sugar),
    ]])

    score = reg_model.predict(scaled_input)[0]
    risk  = clf_model.predict(scaled_input)[0]
    proba = clf_model.predict_proba(scaled_input)[0]

    st.markdown("---")
    st.subheader("📊 Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Progression Score", f"{score:.1f}", help=f"Dataset median: {median_val:.1f}")
    m2.metric("High Risk", f"{proba[1]*100:.1f}%")
    m3.metric("Low Risk",  f"{proba[0]*100:.1f}%")

    if risk == 1:
        st.error("🚨 **HIGH RISK** — Progression likely above median.")
    else:
        st.success("✅ **LOW RISK** — Progression likely below median.")
