import streamlit as st
import joblib
import numpy as np
import os
st.write("### DEBUG — Files visible to app:")
st.write(os.listdir("."))
st.write("Current directory:", os.getcwd())
st.stop()
st.set_page_config(page_title="Diabetes AI", page_icon="🧬", layout="wide")

# ---------- MODERN THEME ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif !important; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}

h1 {
    font-family: 'Poppins', sans-serif !important;
    color: #c4b5fd !important;
    font-weight: 700 !important;
    text-align: center;
    letter-spacing: -1.5px;
    padding-bottom: 10px;
}
h2, h3 { color: #a5b4fc !important; font-weight: 600 !important; }

/* Remove orange — sliders get purple */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #a78bfa !important;
    border: 3px solid #c4b5fd !important;
    box-shadow: 0 0 20px rgba(167, 139, 250, 0.7) !important;
    height: 20px; width: 20px;
}
.stSlider [data-baseweb="slider"] > div > div {
    background: linear-gradient(90deg, #6366f1, #a78bfa) !important;
}
.stSlider label {
    color: #e0e7ff !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}

/* Labels (values shown above slider) */
.stSlider [data-testid="stTickBar"] { display: none; }
.stSlider [data-baseweb="slider"] + div p {
    color: #c4b5fd !important;
    font-weight: 600 !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 14px 40px !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    letter-spacing: 0.5px;
    width: 100%;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 8px 24px rgba(139, 92, 246, 0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 15px 35px rgba(139, 92, 246, 0.6) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255,255,255,0.03);
    padding: 8px; border-radius: 50px;
    justify-content: center;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 50px; padding: 10px 24px;
    color: #a5b4fc !important; font-weight: 600;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #8b5cf6, #6366f1) !important;
    color: #ffffff !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(167, 139, 250, 0.08);
    border: 1.5px solid rgba(167, 139, 250, 0.3);
    border-radius: 20px; padding: 24px;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(167, 139, 250, 0.6);
    transform: translateY(-3px);
}
[data-testid="stMetricValue"] {
    color: #c4b5fd !important;
    font-weight: 700 !important;
    font-size: 36px !important;
}
[data-testid="stMetricLabel"] { color: #a5b4fc !important; }

/* Alerts */
.stAlert { border-radius: 15px !important; border-left: 5px solid !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.6);
    border-right: 1px solid rgba(167, 139, 250, 0.2);
}

/* Caption */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #8b8fb5 !important; text-align: center;
}
</style>
""", unsafe_allow_html=True)
# ---------- END THEME ----------

reg_model  = joblib.load("diabetes_reg_model.joblib")
clf_model  = joblib.load("diabetes_clf_model.joblib")
scaler     = joblib.load("diabetes_scaler.joblib")
median_val = joblib.load("diabetes_median.joblib")

# Feature definitions — realistic medical ranges
FEATURES = {
    'age':              {'label': '🧑 Age',              'range': (20, 80),   'step': 1,   'default': 50,  'mean': 49,  'std': 15,  'unit': 'years'},
    'sex':              {'label': '⚧ Sex (-1=M, 1=F)', 'range': (-1.0, 1.0),'step': 1.0, 'default': 1.0, 'mean': 0,   'std': 1,   'unit': ''},
    'bmi':              {'label': '⚖️ BMI',              'range': (15.0, 45.0),'step': 0.1,'default': 27.0,'mean': 27,  'std': 5,   'unit': 'kg/m²'},
    'blood_pressure':   {'label': '💓 Blood Pressure',  'range': (55, 130),  'step': 1,   'default': 85,  'mean': 85,  'std': 12,  'unit': 'mmHg'},
    'total_cholesterol':{'label': '🧪 Total Cholesterol','range': (80, 320), 'step': 1,   'default': 180, 'mean': 180, 'std': 40,  'unit': 'mg/dL'},
    'ldl':              {'label': '🟡 LDL',             'range': (40, 220),  'step': 1,   'default': 115, 'mean': 115, 'std': 30,  'unit': 'mg/dL'},
    'hdl':              {'label': '🟢 HDL',             'range': (15, 100),  'step': 1,   'default': 50,  'mean': 50,  'std': 15,  'unit': 'mg/dL'},
    'chol_hdl_ratio':   {'label': '📊 Chol/HDL Ratio',  'range': (1.5, 10.0),'step': 0.1, 'default': 4.0, 'mean': 4,   'std': 1.5, 'unit': ''},
    'triglycerides':    {'label': '🔵 Triglycerides',   'range': (30, 400),  'step': 1,   'default': 120, 'mean': 120, 'std': 60,  'unit': 'mg/dL'},
    'blood_sugar':      {'label': '🍬 Blood Sugar',     'range': (55, 250),  'step': 1,   'default': 95,  'mean': 95,  'std': 25,  'unit': 'mg/dL'},
}

with st.sidebar:
    st.markdown("### 🧬 Diabetes AI")
    st.markdown("---")
    st.write("**Models used**")
    st.write("• Linear Regression (score)")
    st.write("• Random Forest (risk)")
    st.markdown("---")
    st.write("**Dataset**")
    st.write("sklearn load_diabetes()")
    st.write("442 patients · 10 features")
    st.markdown("---")
    st.caption("⚠️ Educational demo only")

st.title("🧬 Diabetes Progression & Risk AI")
st.markdown(
    "<p style='text-align:center; color:#a5b4fc; font-size:16px;'>"
    "Enter patient measurements to predict diabetes progression score and risk level."
    "</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🧑 Patient Vitals", "🧪 Blood Panel", "🔬 Predict"])

with tab1:
    st.markdown("#### Basic Measurements")
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider(FEATURES['age']['label'], *FEATURES['age']['range'],
                        FEATURES['age']['default'], FEATURES['age']['step'],
                        help=f"Range: {FEATURES['age']['range']} {FEATURES['age']['unit']}")
        bmi = st.slider(FEATURES['bmi']['label'], *FEATURES['bmi']['range'],
                        FEATURES['bmi']['default'], FEATURES['bmi']['step'],
                        help=f"Range: {FEATURES['bmi']['range']} {FEATURES['bmi']['unit']}")
    with c2:
        sex = st.slider(FEATURES['sex']['label'], *FEATURES['sex']['range'],
                        FEATURES['sex']['default'], FEATURES['sex']['step'])
        blood_pressure = st.slider(FEATURES['blood_pressure']['label'], *FEATURES['blood_pressure']['range'],
                                    FEATURES['blood_pressure']['default'], FEATURES['blood_pressure']['step'],
                                    help=f"Range: {FEATURES['blood_pressure']['range']} {FEATURES['blood_pressure']['unit']}")

with tab2:
    st.markdown("#### Blood Test Results")
    c1, c2 = st.columns(2)
    with c1:
        total_cholesterol = st.slider(FEATURES['total_cholesterol']['label'], *FEATURES['total_cholesterol']['range'],
                                       FEATURES['total_cholesterol']['default'], FEATURES['total_cholesterol']['step'])
        ldl = st.slider(FEATURES['ldl']['label'], *FEATURES['ldl']['range'],
                        FEATURES['ldl']['default'], FEATURES['ldl']['step'])
        hdl = st.slider(FEATURES['hdl']['label'], *FEATURES['hdl']['range'],
                        FEATURES['hdl']['default'], FEATURES['hdl']['step'])
    with c2:
        chol_hdl_ratio = st.slider(FEATURES['chol_hdl_ratio']['label'], *FEATURES['chol_hdl_ratio']['range'],
                                    FEATURES['chol_hdl_ratio']['default'], FEATURES['chol_hdl_ratio']['step'])
        triglycerides = st.slider(FEATURES['triglycerides']['label'], *FEATURES['triglycerides']['range'],
                                   FEATURES['triglycerides']['default'], FEATURES['triglycerides']['step'])
        blood_sugar = st.slider(FEATURES['blood_sugar']['label'], *FEATURES['blood_sugar']['range'],
                                 FEATURES['blood_sugar']['default'], FEATURES['blood_sugar']['step'])

with tab3:
    st.markdown("#### Run Prediction")
    st.write("All inputs captured. Click below to get the AI prediction.")

    if st.button("🔬 Predict Now", type="primary"):
        # Convert realistic values → scaled values the model expects
        vals = {'age': age, 'sex': sex, 'bmi': bmi, 'blood_pressure': blood_pressure,
                'total_cholesterol': total_cholesterol, 'ldl': ldl, 'hdl': hdl,
                'chol_hdl_ratio': chol_hdl_ratio, 'triglycerides': triglycerides,
                'blood_sugar': blood_sugar}

        scaled_input = []
        for key, v in vals.items():
            s = FEATURES[key]
            scaled_input.append((v - s['mean']) / s['std'] * 0.0476)

        X_input = scaler.transform(np.array([scaled_input]))

        score = reg_model.predict(X_input)[0]
        risk  = clf_model.predict(X_input)[0]
        proba = clf_model.predict_proba(X_input)[0]

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("📈 Predicted Score", f"{score:.1f}",
                      help=f"Dataset median: {median_val:.1f}")
        with c2:
            if risk == 1:
                st.error("### 🔴 HIGH RISK")
            else:
                st.success("### 🟢 LOW RISK")

        st.markdown("#### Confidence")
        st.progress(float(proba[1]), text=f"🔴 High Risk: {proba[1]*100:.1f}%")
        st.progress(float(proba[0]), text=f"🟢 Low Risk:  {proba[0]*100:.1f}%")

st.markdown("---")
st.caption("Built with Streamlit · sklearn · AI in Healthcare Bootcamp 2026")
