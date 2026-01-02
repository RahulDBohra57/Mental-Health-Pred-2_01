# ============================================================
# Mental Health Cluster Insight Tool
# Final Version (Stable, Clinical-Grade, Risk-Aware)
# ============================================================

import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# ============================================================
# Load Model Artifacts (No assumptions)
# ============================================================

@st.cache_resource
def load_artifacts():
    mca = joblib.load("mca_transformer.joblib")
    cluster_models = joblib.load("risk_band_cluster_models.joblib")
    severity_map = joblib.load("severity_map.joblib")
    return mca, cluster_models, severity_map

mca, cluster_models, severity_map = load_artifacts()

# ============================================================
# Feature Definitions (MUST MATCH TRAINING)
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
    "Coping_Struggles",
    "Work_Interest",
    "Social_Weakness"
]

# ============================================================
# UI Question Configuration
# ============================================================

question_config = {
    "family_history": {
        "question": "Is there a history of mental health concerns in your family?",
        "options": ["Yes", "No"]
    },
    "treatment": {
        "question": "Are you currently receiving any mental health support or treatment?",
        "options": ["Yes", "No"]
    },
    "Growing_Stress": {
        "question": "How would you describe your stress levels recently?",
        "options": ["Manageable", "Elevated", "Overwhelming"]
    },
    "Changes_Habits": {
        "question": "Have you noticed changes in your daily habits (sleep, appetite, routine)?",
        "options": ["No noticeable changes", "Some changes", "Significant changes"]
    },
    "Mood_Swings": {
        "question": "How frequently do you experience mood swings or emotional ups and downs?",
        "options": ["Rarely", "Sometimes", "Often"]
    },
    "Coping_Struggles": {
        "question": "How well do you feel you are coping with everyday challenges?",
        "options": ["Coping well", "Struggling at times", "Struggling most of the time"]
    },
    "Work_Interest": {
        "question": "How engaged do you feel with your work or daily responsibilities lately?",
        "options": ["Highly engaged", "Somewhat engaged", "Not engaged"]
    },
    "Social_Weakness": {
        "question": "How socially connected do you feel compared to your usual self?",
        "options": ["As connected as usual", "Slightly less connected", "Much less connected"]
    }
}

# ============================================================
# MDI & Risk Logic
# ============================================================

def calculate_mdi(user_input):
    return sum(severity_map.get(user_input[col], 1) for col in core_symptoms)

def assign_risk_band(mdi):
    if mdi >= 8:
        return "High"
    elif mdi >= 4:
        return "Moderate"
    else:
        return "Low"

# ============================================================
# Diagnosis & Suggestions
# ============================================================

diagnosis_map = {
    "Low": "Your responses suggest stable emotional well-being with healthy coping patterns.",
    "Moderate": "Your responses indicate ongoing stress that may be affecting balance and daily energy.",
    "High": "Your responses reflect significant emotional strain that may be overwhelming your current coping capacity."
}

suggestions_map = {
    "Low": [
        "Maintain consistent sleep and daily routines.",
        "Continue activities that help you relax or feel fulfilled.",
        "Stay socially connected with people you trust.",
        "Practice occasional self-reflection or journaling.",
        "Maintain healthy work–life boundaries.",
        "Respond early when stress levels begin to rise."
    ],
    "Moderate": [
        "Break daily tasks into smaller, manageable steps.",
        "Schedule at least one restorative break each day.",
        "Reduce non-essential commitments temporarily.",
        "Engage in light physical activity such as walking.",
        "Practice grounding or breathing exercises.",
        "Talk openly with a trusted friend or family member.",
        "Re-establish consistent sleep and meal routines."
    ],
    "High": [
        "Prioritize rest and reduce mental overload wherever possible.",
        "Seek support from a trusted person instead of coping alone.",
        "Use grounding techniques such as slow breathing or sensory focus.",
        "Avoid major decisions while feeling emotionally overwhelmed.",
        "Create predictable daily structure using small routines.",
        "Limit exposure to unnecessary stressors.",
        "Consider reaching out to a mental health professional.",
        "Spend time in calming environments."
    ]
}

# ============================================================
# Cluster Prediction (Risk-Aware)
# ============================================================

def predict_cluster(user_input):
    mdi = calculate_mdi(user_input)
    risk_band = assign_risk_band(mdi)

    df = pd.DataFrame([user_input]).astype(str)
    X_mca = mca.transform(df[core_symptoms]).fillna(0)
    X_mca.iloc[:, :3] *= 2

    model = cluster_models[risk_band]
    cluster_id = int(model.predict(X_mca)[0])

    return mdi, risk_band, cluster_id

# ============================================================
# PDF Generator
# ============================================================

def generate_pdf(user_name, risk_band, diagnosis, suggestions):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    y = 750

    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, y, "Mental Well-Being Report")
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

    c.setFont("Helvetica", 12)
    c.drawString(50, y, diagnosis)
    y -= 30

    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Recommended Activities")
    y -= 20

    c.setFont("Helvetica", 11)
    for s in suggestions:
        c.drawString(60, y, f"- {s}")
        y -= 15

    c.setFont("Helvetica", 9)
    c.drawString(50, 40, "Disclaimer: This is not a medical diagnosis.")
    c.save()

    buffer.seek(0)
    return buffer

# ============================================================
# STREAMLIT UI
# ============================================================

st.markdown("<h1 style='text-align:center;'>Mental Health Cluster Insight Tool</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

user_name = st.text_input("Enter your name (optional)")

user_input = {}
for feature in features:
    cfg = question_config[feature]
    user_input[feature] = st.selectbox(cfg["question"], cfg["options"])

if st.button("Generate My Well-Being Insights"):
    mdi, risk_band, cluster_id = predict_cluster(user_input)

    color_map = {
        "Low": "#2ecc71",
        "Moderate": "#f1c40f",
        "High": "#e74c3c"
    }

    st.markdown(
        f"""
        <div style="padding:12px;border-radius:6px;
        background-color:{color_map[risk_band]};
        color:black;font-weight:bold;">
        Risk Level: {risk_band}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Assessment Summary")
    st.write(diagnosis_map[risk_band])

    st.subheader("Suggested Activities")
    for s in suggestions_map[risk_band]:
        st.write(f"• {s}")

    pdf = generate_pdf(
        user_name,
        risk_band,
        diagnosis_map[risk_band],
        suggestions_map[risk_band]
    )

    st.download_button(
        "Download My Wellness Report (PDF)",
        data=pdf,
        file_name="Wellness_Report.pdf",
        mime="application/pdf"
    )

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<small><b>Disclaimer:</b> This tool provides general well-being insights only. "
    "It is not a medical diagnosis.</small>",
    unsafe_allow_html=True
)
