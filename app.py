# ============================================================
# Mental Health Cluster Insight Tool (Clinical-Grade)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from prince import MCA
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
from datetime import datetime

# ============================================================
# Load Trained Artifacts
# ============================================================

@st.cache_resource
def load_artifacts():
    mca = joblib.load("mca_transformer.joblib")
    cluster_models = joblib.load("risk_band_cluster_models.joblib")
    severity_map = joblib.load("severity_map.joblib")
    return mca, cluster_models, severity_map

mca, cluster_models, severity_map, ui_categories = load_artifacts()

# ============================================================
# Feature Definitions
# ============================================================

features = [
    "family_history",
    "treatment",
    "Growing_Stress",
    "Changes_Habits",
    "Mood_Swings",
    "Coping_Struggles",
    "Work_Interest",
    "Social_Weakness"
]

core_symptoms = [
    "Growing_Stress",
    "Changes_Habits",
    "Mood_Swings",
    "Coping_Struggles"
]

functional_impact = [
    "Work_Interest",
    "Social_Weakness"
]

# ============================================================
# Friendly UI Labels
# ============================================================

friendly_labels = {
    "family_history": "Does your family have a history of mental health concerns?",
    "treatment": "Are you currently undergoing any mental health treatment?",
    "Growing_Stress": "Do you feel your stress levels have been increasing recently?",
    "Changes_Habits": "Have you noticed recent changes in your habits (sleep, diet, routines)?",
    "Mood_Swings": "Do you experience noticeable mood swings?",
    "Coping_Struggles": "Do you struggle to cope with everyday challenges?",
    "Work_Interest": "How interested are you in your work lately?",
    "Social_Weakness": "Do you feel socially withdrawn or less engaged than usual?"
}

# ============================================================
# Risk Band Logic
# ============================================================

def calculate_mdi(user_input):
    score = 0

    for col in core_symptoms + functional_impact:
        value = user_input.get(col)
        score += severity_map.get(value, 0)

    # Reverse work interest (low interest = higher distress)
    score += (3 - severity_map.get(user_input.get("Work_Interest"), 0))

    return score


def assign_risk_band(mdi):
    if mdi >= 10:
        return "High"
    elif mdi >= 6:
        return "Moderate"
    else:
        return "Low"

# ============================================================
# Cluster Interpretation
# ============================================================

cluster_interpretations = {
    "Low": {
        0: ("Stable & Balanced",
            "You show emotional stability and healthy coping patterns."),
        1: ("Mild Stress but Adaptive",
            "You experience occasional stress but demonstrate resilience.")
    },
    "Moderate": {
        0: ("Emotional Variability",
            "You may experience mood fluctuations or inconsistent routines."),
        1: ("Stress Accumulation",
            "Stressors are building and may affect daily functioning."),
        2: ("Emerging Burnout Pattern",
            "Signs of fatigue or disengagement are becoming noticeable.")
    },
    "High": {
        0: ("High Emotional Dysregulation",
            "Strong emotional distress and difficulty regulating mood."),
        1: ("Withdrawal & Burnout",
            "Reduced engagement and emotional exhaustion are present."),
        2: ("Acute Stress Overload",
            "High stress levels may be overwhelming coping capacity.")
    }
}

# ============================================================
# Prediction Function (Clinical-Grade)
# ============================================================

def predict_cluster(user_input):
    mdi = calculate_mdi(user_input)
    risk_band = assign_risk_band(mdi)

    df = pd.DataFrame([user_input]).astype(str)
    X_mca = mca.transform(df[core_symptoms + functional_impact])

    # Weight symptom-heavy components
    X_weighted = X_mca.copy()
    X_weighted.iloc[:, :3] = X_weighted.iloc[:, :3] * 2

    model = cluster_models[risk_band]
    cluster_id = int(model.predict(X_weighted)[0])

    return mdi, risk_band, cluster_id

# ============================================================
# PDF REPORT GENERATOR
# ============================================================

def generate_pdf(user_name, mdi, risk_band, title, description, responses):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 50
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, y, "Mental Well-Being Insight Report")
    y -= 40

    c.setFont("Helvetica", 12)
    if user_name:
        c.drawString(50, y, f"Prepared For: {user_name}")
        y -= 20

    now = datetime.now().strftime("%d/%m/%Y, %I:%M %p")
    c.drawString(50, y, f"Date Generated: {now}")
    y -= 30

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, f"Risk Level: {risk_band}")
    y -= 20

    c.drawString(50, y, f"Cluster: {title}")
    y -= 30

    c.setFont("Helvetica", 12)
    c.drawString(50, y, description)
    y -= 40

    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Your Responses")
    y -= 20

    c.setFont("Helvetica", 11)
    for k, v in responses.items():
        c.drawString(60, y, f"{k.replace('_',' ').title()}: {v}")
        y -= 15

    c.setFont("Helvetica", 9)
    c.drawString(
        50, 40,
        "Disclaimer: This report provides general well-being insights and is not a medical diagnosis."
    )

    c.save()
    buffer.seek(0)
    return buffer

# ============================================================
# STREAMLIT UI
# ============================================================

st.markdown("<h1 style='text-align:center;'>Mental Health Cluster Insight Tool</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:gray;'>A supportive, non-diagnostic emotional well-being assessment.</p>",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

st.subheader("Your Details")
user_name = st.text_input("Enter your name (optional)")

st.subheader("Your Responses")
user_input = {}

for feature in features:
    options = ui_categories[feature].dropna().tolist()
    user_input[feature] = st.selectbox(friendly_labels[feature], options)

st.markdown("<hr>", unsafe_allow_html=True)

if st.button("Generate My Well-Being Insights"):
    mdi, risk_band, cluster_id = predict_cluster(user_input)

    title, description = cluster_interpretations[risk_band][cluster_id]

    st.success(f"Risk Level: {risk_band}")
    st.subheader(title)
    st.write(description)

    st.caption(
        "This assessment combines symptom severity and pattern-based clustering "
        "to provide responsible and supportive insights."
    )

    pdf = generate_pdf(
        user_name,
        mdi,
        risk_band,
        title,
        description,
        user_input
    )

    st.download_button(
        "Download My Wellness Report (PDF)",
        data=pdf,
        file_name="Wellness_Report.pdf",
        mime="application/pdf"
    )

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<small><b>Disclaimer:</b> This tool does not provide medical advice or diagnosis. "
    "If you are experiencing emotional distress, please consult a qualified mental health professional.</small>",
    unsafe_allow_html=True
)
