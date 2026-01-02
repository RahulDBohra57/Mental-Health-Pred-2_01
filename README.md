# 🧠 Mental Health Cluster Insight Tool

A **clinical-grade, ethical, and explainable mental well-being assessment tool** built using unsupervised machine learning and rule-based severity logic. The application helps users understand **patterns** in their emotional well-being — **not** diagnose medical conditions.

---

## 📌 Project Objective

Traditional clustering models often fail in mental-health contexts because they **average emotional distress**, leading to false reassurance (e.g., severe cases classified as “mild stress”).

This project solves that problem by using a **hybrid approach**:

1. **Symptom severity scoring** (Mental Distress Index – MDI)
2. **Risk band assignment** (Low / Moderate / High)
3. **Clustering within risk bands only**

This ensures that **high-distress users are never mixed with low-distress profiles**, making the system safer, more interpretable, and clinically defensible.

---

## 🧩 Key Features

- 🧠 Mental Distress Index (MDI) for severity awareness
- 🛡 Risk-band first modeling (clinical safety layer)
- 📊 MCA-based clustering for categorical data
- 🧬 Separate handling of symptoms vs functional impact
- 📄 Downloadable PDF well-being report
- ⚖️ Ethical, non-diagnostic framing
- 🧑‍⚕️ Interview & academic-ready architecture

---

## 📂 Project Structure

```
├── app.py                         # Streamlit application
├── Mental Health Dataset.csv      # Training dataset
├── Mental Health Analysis App.ipynb  # Model training notebook
├── mca_transformer.joblib         # Trained MCA transformer
├── risk_band_cluster_models.joblib # Cluster models per risk band
├── severity_map.joblib            # Severity encoding dictionary
├── ui_categories.csv              # UI dropdown options
├── README.md                      # Project documentation
```

---

## 📊 Dataset Overview

The dataset consists of **self-reported mental health indicators**, all categorical in nature.

### Feature Groups

#### 1. Contextual Variables

- `family_history`
- `treatment`

> Used for background context only (not severity drivers)

#### 2. Core Symptom Indicators

- `Growing_Stress`
- `Changes_Habits`
- `Mood_Swings`
- `Coping_Struggles`

> Primary indicators of emotional distress

#### 3. Functional Impact Indicators

- `Work_Interest`
- `Social_Weakness`

> Strongest signals of depression-like patterns

---

## 🧠 Mental Distress Index (MDI)

To overcome the limitations of pure clustering, the project introduces a **Mental Distress Index**.

### Severity Encoding

| Response | Severity Score |
| -------- | -------------- |
| No       | 0              |
| Low      | 1              |
| Medium   | 2              |
| High     | 3              |
| Yes      | 2              |

### MDI Formula

The MDI is computed by combining:

- Core symptom severity
- Functional impairment
- Inverse work interest (low interest → higher distress)

> Higher MDI = higher emotional distress

---

## 🛡 Risk Banding (Clinical Safety Layer)

Before clustering, each user is assigned a **risk band** based on MDI:

| Risk Band | Interpretation                         |
| --------- | -------------------------------------- |
| Low       | Emotionally stable / mild stress       |
| Moderate  | Emerging or persistent stress patterns |
| High      | Significant emotional distress         |

This prevents **severe cases from being averaged into mild clusters**.

---

## 🔍 Modeling Approach

### Why Not Pure KMeans?

- KMeans optimizes distance, **not severity**
- Extreme emotional responses get pulled toward centroids
- Dangerous in mental-health applications

### Final Architecture

```
User Input
   ↓
Mental Distress Index (MDI)
   ↓
Risk Band Assignment
   ↓
MCA Transformation
   ↓
Weighted Clustering (per risk band)
   ↓
Clinically Interpretable Cluster Output
```

---

## 🧪 Algorithms Used

- **Multiple Correspondence Analysis (MCA)**

  - For dimensionality reduction on categorical data

- **KMeans Clustering (per risk band)**

  - Applied only within Low / Moderate / High bands
  - Prevents centroid dilution

- **Rule-based Severity Logic**
  - Ensures ethical and safe outputs

---

## 🖥 Application (Streamlit)

The Streamlit app allows users to:

- Answer 8 well-being questions
- Receive:
  - Risk level (Low / Moderate / High)
  - Cluster-based emotional pattern
  - Supportive interpretation
- Download a **PDF wellness report**

> The app **does not diagnose** and clearly communicates this to users.

---

## 📄 PDF Report Contents

- User name (optional)
- Date & time generated
- Risk level
- Cluster title & interpretation
- User responses
- Ethical disclaimer

---

## ⚠️ Ethical Disclaimer

This tool:

- ❌ Does NOT diagnose mental illness
- ❌ Is NOT a replacement for professional care
- ✅ Provides pattern-based well-being insights only

Users experiencing emotional distress are advised to consult a qualified mental health professional.

---

## 🎯 Why This Project Is Strong

- Clinically aware design
- Hybrid ML + rules (industry standard)
- Explainable & interpretable
- Avoids false reassurance
- Suitable for interviews, viva, and production demos

---

## 🚀 Future Enhancements

- Confidence / distance-to-centroid score
- Crisis escalation messaging
- Longitudinal tracking
- Model validation with labeled data
- Deployment on cloud (AWS / GCP)

---

## 👤 Author Notes

This project demonstrates how **responsible AI design** is essential when applying machine learning to sensitive domains like mental health.

> Accuracy is important — **safety is non-negotiable**.
