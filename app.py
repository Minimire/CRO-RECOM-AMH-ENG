# ==============================================================================
# PROJECT: Intelligent Crop Recommendation System for Precision Agriculture
# COMPONENT: Frontend Streamlit Web User Interface App (app.py)
# DEPLOYMENT TARGET: Streamlit Community Cloud (RAM Optimized < 1GB Layer)
# ==============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Set professional layout configurations
st.set_page_config(
    page_title="AI Crop Recommendation System",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Global layout styling optimizations
st.markdown("""
<style>
.main { background-color: #f9fbf9; }
.stButton>button { width: 100%; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)  # <--- Fixed parameter name


# Define static fallback range guards (derived directly from structural training limits)
BOUNDARIES = {
    'N': {'min': 0.0, 'max': 140.0, 'default': 50.0, 'step': 1.0, 'unit': 'mg/kg', 'label': 'Nitrogen (N)'},
    'P': {'min': 5.0, 'max': 145.0, 'default': 40.0, 'step': 1.0, 'unit': 'mg/kg', 'label': 'Phosphorus (P)'},
    'K': {'min': 5.0, 'max': 205.0, 'default': 40.0, 'step': 1.0, 'unit': 'mg/kg', 'label': 'Potassium (K)'},
    'temperature': {'min': 8.0, 'max': 45.0, 'default': 25.0, 'step': 0.1, 'unit': '°C', 'label': 'Ambient Temperature'},
    'humidity': {'min': 14.0, 'max': 100.0, 'default': 70.0, 'step': 0.1, 'unit': '%', 'label': 'Relative Humidity'},
    'ph': {'min': 3.5, 'max': 10.0, 'default': 6.5, 'step': 0.1, 'unit': 'pH', 'label': 'Soil pH Scale'},
    'rainfall': {'min': 20.0, 'max': 300.0, 'default': 100.0, 'step': 0.5, 'unit': 'mm', 'label': 'Seasonal Precipitation Depth'}
}

@st.cache_resource(show_spinner="Loading trained production model systems...")
def load_production_artifacts():
    """Loads and caches the compound model bundle dictionary to save memory."""
    artifact_path = 'crop_recommendation_system_bundle.pkl'
    if os.path.exists(artifact_path):
        try:
            return joblib.load(artifact_path)
        except Exception as e:
            st.error(f"Error loading system binaries: {str(e)}")
            return None
    else:
        return None

# Initializing pipeline bundle matrix
artifacts = load_production_artifacts()

# App Header Layout
st.title("🌾 Intelligent Crop Recommendation System")
st.subheader("Data-Driven Precision Agriculture Decision Support Engine")

# ==============================================================================
# SIDEBAR SAFETY & RESPONSIBLE AI ADVISORY NOTICE
# ==============================================================================
with st.sidebar:
    st.header("⚖️ Responsible AI Advisory")
    st.info(
        "**Notice to System Operators:**\n"
        "This tool serves strictly as an agronomic decision-support system to guide strategic agricultural planning. "
        "It provides macro-level guidance based on specific soil and climate training attributes but does not fully replace physical, "
        "localized scientific diagnostic evaluation.\n\n"
        "**Real-World Operating Disclaimers:**\n"
        "1. Does not map soil mechanical constraints (e.g., heavy clay profiles).\n"
        "2. Predictions assume baseline crop resource equity (e.g., standard fertilizer availability).\n"
        "3. Highly volatile microclimatic shock anomalies (e.g., unexpected early frosts) escape macro tracking."
    )
    st.write("---")
    st.caption("FUSE AI-201 Capstone Prototype — Team Group 11")

# Main Interactive User Formulation Workspace
st.markdown("### Step 1: Input Localized Soil Nutrients & Meteorological Attributes")
st.write("Adjust the environmental dimensions below to generate an agronomic recommendation vector.")

col1, col2 = st.columns(2)
input_data = {}

with col1:
    st.markdown("**Soil Nutrient Metrics**")
    for field in ['N', 'P', 'K']:
        cfg = BOUNDARIES[field]
        input_data[field] = st.slider(
            label=f"{cfg['label']} ({cfg['unit']})",
            min_value=cfg['min'],
            max_value=cfg['max'],
            value=cfg['default'],
            step=cfg['step']
        )
        
    st.markdown("**Soil Characteristics**")
    cfg = BOUNDARIES['ph']
    input_data['ph'] = st.slider(
        label=f"{cfg['label']} ({cfg['unit']})",
        min_value=cfg['min'],
        max_value=cfg['max'],
        value=cfg['default'],
        step=cfg['step']
    )

with col2:
    st.markdown("**Microclimate Ambient Vectors**")
    for field in ['temperature', 'humidity', 'rainfall']:
        cfg = BOUNDARIES[field]
        input_data[field] = st.slider(
            label=f"{cfg['label']} ({cfg['unit']})",
            min_value=cfg['min'],
            max_value=cfg['max'],
            value=cfg['default'],
            step=cfg['step']
        )

st.write("---")

# ==============================================================================
# INFERENCE LOGIC PIPELINE EXECUTION
# ==============================================================================
if st.button("Generate Recommendation Engine Vectors"):
    ordered_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    input_vector = np.array([[input_data[f] for f in ordered_features]])
    
    if artifacts is not None:
        try:
            model = artifacts['production_model']
            encoder = artifacts['label_encoder']
            
            # Predict and inverse-transform label target back to crop text string
            encoded_prediction = model.predict(input_vector)
            recommended_crop = encoder.inverse_transform(encoded_prediction)[0]
            
            st.success(f"### 🎯 Optimal Crop Selected: **{recommended_crop.upper()}**")
            
            if hasattr(model, "predict_proba"):
                prob_matrix = model.predict_proba(input_vector)
                confidence = np.max(prob_matrix)
                st.metric(label="Inference Confidence Probability Value", value=f"{confidence:.2%}")
                if confidence < 0.50:
                    st.warning("⚠️ **Low Confidence Support Alert:** The input measurements map into highly ambiguous boundaries across crop clusters. Verify parameters carefully.")
            
            st.balloons()
            
        except Exception as e:
            st.error(f"Inference pipeline runtime failure: {str(e)}")
    else:
        st.info("🤖 **Mock Prototype Inference Mode Enabled**")
        if input_data['rainfall'] > 150:
            mock_crop = "Rice"
        elif input_data['K'] > 150:
            mock_crop = "Grapes"
        else:
            mock_crop = "Maize"
        st.success(f"### 🎯 Optimal Crop Selected (Fallback Prototype Rule): **{mock_crop.upper()}**")

# ==============================================================================
# MODEL INTERPRETABILITY (EXPLAINABLE AI SECTION)
# ==============================================================================
st.write("---")
st.markdown("### 📊 Model Interpretability & Explainable AI (XAI)")

with st.expander("See how the AI model calculates its decisions (SHAP Global Evaluation Framework)"):
    st.markdown(
        "To achieve a robust production deployment that doesn't breach the **Streamlit 1 GB RAM limit**, "
        "the model's global game-theory behaviors have been pre-rendered below via **SHAP (Shapley Additive Explanations)** TreeExplainers."
    )
    
    shap_image = 'shap_summary_global.png'
    if os.path.exists(shap_image):
        st.image(
            shap_image, 
            caption="SHAP Global Attribute Feature Rankings: How atmospheric conditions and soil components drive final recommendations.",
            use_container_width=True
        )
    else:
        st.warning("Pre-rendered SHAP graphical visualization file ('shap_summary_global.png') missing from tracking path.")
        st.markdown("""
        **Global Structural Feature Impacts Explained (SHAP Tree Weights):**
        *   **Relative Humidity & Rainfall:** Dominate primary branches to segment broad crop archetypes (e.g., Arid vs. Saturated zones).
        *   **Potassium (K):** Functions as a critical non-linear logic selector isolating specialty varieties (e.g., Grapes & Apples).
        *   **Nitrogen & Phosphorus:** Act as micro-level classification adjusters across general grain families.
        """)
