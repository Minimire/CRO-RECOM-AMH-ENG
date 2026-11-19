import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Page configuration and styling
# ------------------------------------------------------------
st.set_page_config(
    page_title="Crop Recommendation | የሰብል ምክረ ሃሳብ",
    page_icon="🌱",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Ethiopic:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton, .stSelectbox, .stNumberInput {
    font-family: 'Noto Sans Ethiopic', sans-serif;
}
.main-title {font-size: 2rem; font-weight: 700; margin-bottom: .2rem;}
.sub-title {color: #64748b; margin-bottom: 1.2rem;}
.result-card {
    padding: 1rem 1.2rem;
    border: 1px solid rgba(128,128,128,.35);
    border-radius: 12px;
    margin: .5rem 0 1rem 0;
}
.small-note {font-size: .88rem; color: #64748b;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Translations
# ------------------------------------------------------------
TEXT = {
    "English": {
        "title": "🌱 Intelligent Crop Recommendation System",
        "subtitle": "Machine-learning support for crop selection using soil and climate conditions.",
        "language": "Language / ቋንቋ",
        "inputs": "Enter soil and climate information",
        "N": "Nitrogen (N)",
        "P": "Phosphorus (P)",
        "K": "Potassium (K)",
        "temperature": "Temperature (°C)",
        "humidity": "Relative humidity (%)",
        "ph": "Soil pH",
        "rainfall": "Rainfall (mm)",
        "recommend": "Recommend a crop",
        "result": "Recommended crop",
        "confidence": "Model confidence",
        "explanation": "Explainable AI (XAI)",
        "importance": "Global feature importance",
        "importance_note": "This chart shows which input features the Random Forest model uses most across its trees. It does not prove that a feature causes a crop to grow.",
        "shap_title": "Local SHAP explanation",
        "shap_note": "SHAP explains how the input values influenced this model's score for the recommended crop. Positive values increase the score; negative values decrease it.",
        "shap_unavailable": "A SHAP explanation could not be generated in this deployment. The global feature-importance chart is still available.",
        "advisory": "Responsible-use advisory",
        "advisory_text": "This result is a machine-learning recommendation based on the dataset and entered values. It is not a substitute for soil laboratory testing, local agronomic expertise, weather forecasts, or advice from agricultural extension officers. Check local conditions before making planting decisions.",
        "out_of_range": "One or more values are outside the broad ranges represented in the training dataset. Treat the prediction with extra caution.",
        "model_error": "The model could not be loaded. Check that the model file is present in the repository and that the bundle contains the required keys.",
        "footer": "FUSE AI-201 | Intelligent Crop Recommendation System",
        "no_prediction": "Enter the values and select Recommend a crop to see a result.",
        "top_crops": "Other likely crops",
        "invalid": "Please enter valid numeric values.",
        "crop_names": "Crop names are shown in English because the model's original class labels are English."
    },
    "አማርኛ": {
        "title": "🌱 ብልህ የሰብል ምክረ ሃሳብ ስርዓት",
        "subtitle": "በአፈርና በአየር ሁኔታ መረጃ ላይ ተመስርቶ ሰብል ለመምረጥ የማሽን መማር ድጋፍ።",
        "language": "ቋንቋ / Language",
        "inputs": "የአፈርና የአየር ሁኔታ መረጃ ያስገቡ",
        "N": "ናይትሮጅን (N)",
        "P": "ፎስፈረስ (P)",
        "K": "ፖታስየም (K)",
        "temperature": "የአየር ሙቀት (°C)",
        "humidity": "አንጻራዊ እርጥበት (%)",
        "ph": "የአፈር pH",
        "rainfall": "የዝናብ መጠን (ሚ.ሜ)",
        "recommend": "ሰብል ይመክሩ",
        "result": "የተመከረው ሰብል",
        "confidence": "የሞዴሉ እምነት መጠን",
        "explanation": "ሊገለጽ የሚችል ሰው ሰራሽ ብልህነት (XAI)",
        "importance": "አጠቃላይ የግቤት ባህሪያት አስፈላጊነት",
        "importance_note": "ይህ ሰንጠረዥ የRandom Forest ሞዴሉ በውሳኔዎቹ ውስጥ የትኞቹን የግቤት ባህሪያት በብዛት እንደሚጠቀም ያሳያል። ይህ ባህሪ ሰብል እንዲያድግ መንስኤ መሆኑን አያረጋግጥም።",
        "shap_title": "የSHAP የግል ማብራሪያ",
        "shap_note": "SHAP የገቡት እሴቶች በተመከረው ሰብል የሞዴል ውጤት ላይ እንዴት ተጽዕኖ እንዳሳደሩ ያብራራል። አዎንታዊ እሴት ውጤቱን ይጨምራል፤ አሉታዊ እሴት ይቀንሳል።",
        "shap_unavailable": "በዚህ ማስኬጃ አካባቢ የSHAP ማብራሪያ ማዘጋጀት አልተቻለም። አጠቃላይ የባህሪያት አስፈላጊነት ሰንጠረዥ ግን ይገኛል።",
        "advisory": "የኃላፊነት አጠቃቀም ማስጠንቀቂያ",
        "advisory_text": "ይህ ውጤት በዳታሴቱና በገቡት እሴቶች ላይ የተመሰረተ የማሽን መማር ምክረ ሃሳብ ነው። የአፈር ላቦራቶሪ ምርመራን፣ የአካባቢ የግብርና ባለሙያ እውቀትን፣ የአየር ትንበያን ወይም የግብርና ማስፋፊያ ባለሙያዎችን ምክር አይተካም። ከመዝራትዎ በፊት የአካባቢዎን ሁኔታ ያረጋግጡ።",
        "out_of_range": "አንድ ወይም ከዚያ በላይ እሴቶች ከስልጠና ዳታሴቱ ሰፊ ክልል ውጭ ናቸው። ውጤቱን በተጨማሪ ጥንቃቄ ይጠቀሙበት።",
        "model_error": "ሞዴሉን መጫን አልተቻለም። የሞዴሉ ፋይል በሪፖዚቶሪው ውስጥ መኖሩንና ትክክለኛ ቁልፎችን መያዙን ያረጋግጡ።",
        "footer": "FUSE AI-201 | ብልህ የሰብል ምክረ ሃሳብ ስርዓት",
        "no_prediction": "ውጤት ለማየት እሴቶቹን ያስገቡና ሰብል ይመክሩ የሚለውን ይጫኑ።",
        "top_crops": "ሌሎች ሊመከሩ የሚችሉ ሰብሎች",
        "invalid": "እባክዎ ትክክለኛ የቁጥር እሴቶችን ያስገቡ።",
        "crop_names": "የሞዴሉ የመጀመሪያ የሰብል ስሞች በእንግሊዝኛ ስለሆኑ እዚህም በእንግሊዝኛ ይታያሉ።"
    }
}

CROP_AMHARIC = {
    "rice": "ሩዝ",
    "maize": "በቆሎ",
    "chickpea": "ሽምብራ",
    "kidneybeans": "ቀይ ባቄላ",
    "pigeonpeas": "ክክ",
    "mothbeans": "ሞት ባቄላ",
    "mungbean": "ማሽ",
    "blackgram": "ጥቁር ባቄላ",
    "lentil": "ምስር",
    "pomegranate": "ሮማን",
    "banana": "ሙዝ",
    "mango": "ማንጎ",
    "grapes": "ወይን",
    "watermelon": "ሐብሐብ",
    "muskmelon": "ካንታሎፕ",
    "apple": "አፕል",
    "orange": "ብርቱካን",
    "papaya": "ፓፓያ",
    "coconut": "ኮኮናት",
    "cotton": "ጥጥ",
    "jute": "ጁት",
    "coffee": "ቡና"
}

# ------------------------------------------------------------
# Load trained model bundle
# ------------------------------------------------------------
MODEL_PATH = "crop_recommendation_system_bundle.pkl"

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model file: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise TypeError("The model bundle must be a dictionary.")
    if "production_model" not in bundle or "label_encoder" not in bundle:
        raise KeyError("Bundle must contain production_model and label_encoder.")
    return bundle["production_model"], bundle["label_encoder"]

try:
    model, label_encoder = load_model(MODEL_PATH)
except Exception as exc:
    st.error(TEXT["English"]["model_error"])
    st.code(str(exc))
    st.stop()

# ------------------------------------------------------------
# Interface
# ------------------------------------------------------------
language = st.sidebar.selectbox(
    "Language / ቋንቋ",
    ["English", "አማርኛ"],
    index=0
)
t = TEXT[language]

st.markdown(f'<div class="main-title">{t["title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">{t["subtitle"]}</div>', unsafe_allow_html=True)

st.header(t["inputs"])

# Broad input bounds for usability; not a guarantee of agronomic suitability.
c1, c2 = st.columns(2)
with c1:
    N = st.number_input(t["N"], min_value=0.0, max_value=300.0, value=90.0, step=1.0)
    P = st.number_input(t["P"], min_value=0.0, max_value=300.0, value=42.0, step=1.0)
    K = st.number_input(t["K"], min_value=0.0, max_value=300.0, value=43.0, step=1.0)
    temperature = st.number_input(t["temperature"], min_value=-10.0, max_value=60.0, value=25.0, step=0.5)
with c2:
    humidity = st.number_input(t["humidity"], min_value=0.0, max_value=100.0, value=80.0, step=0.5)
    ph = st.number_input(t["ph"], min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    rainfall = st.number_input(t["rainfall"], min_value=0.0, max_value=2000.0, value=200.0, step=1.0)

predict_clicked = st.button(t["recommend"], type="primary", use_container_width=True)

if predict_clicked:
    input_data = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    )

    try:
        prediction_encoded = int(model.predict(input_data)[0])
        predicted_crop = str(label_encoder.inverse_transform([prediction_encoded])[0])

        st.markdown("---")
        st.subheader(t["result"])
        shown_crop = CROP_AMHARIC.get(predicted_crop, predicted_crop) if language == "አማርኛ" else predicted_crop
        st.markdown(
            f'<div class="result-card"><h2>{shown_crop}</h2><p>{predicted_crop}</p></div>',
            unsafe_allow_html=True
        )

        # Confidence and alternative predictions
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            confidence = float(np.max(probabilities)) * 100
            st.metric(t["confidence"], f"{confidence:.1f}%")

            top_indices = np.argsort(probabilities)[::-1][1:4]
            alternatives = []
            for idx in top_indices:
                crop = str(label_encoder.inverse_transform([int(idx)])[0])
                crop_display = CROP_AMHARIC.get(crop, crop) if language == "አማርኛ" else crop
                alternatives.append({
                    "Crop": crop_display,
                    "Probability (%)": round(float(probabilities[idx]) * 100, 2)
                })
            if alternatives:
                st.write(f"**{t['top_crops']}**")
                st.dataframe(pd.DataFrame(alternatives), use_container_width=True, hide_index=True)

        # Warn for values outside the approximate training-data envelope.
        training_bounds = {
            "N": (0, 140), "P": (5, 145), "K": (5, 205),
            "temperature": (8, 44), "humidity": (14, 100),
            "ph": (3.5, 10), "rainfall": (20, 300)
        }
        values = {
            "N": N, "P": P, "K": K, "temperature": temperature,
            "humidity": humidity, "ph": ph, "rainfall": rainfall
        }
                outside = [
            key for key, value in values.items()
            if not (training_bounds[key][0] <= value <= training_bounds[key][1])
        ]

        if outside:
            st.warning(
                t["out_of_range"] + " (" + ", ".join(outside) + ")"
            )

        features = [
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall"
        ]

        # SHAP explanation
        # Explainability: global feature importance
        st.header(t["explanation"])

        if hasattr(model, "feature_importances_"):
            st.subheader(t["importance"])
            importance_df = pd.DataFrame({
                "Feature": ["N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall"],
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=True)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(importance_df["Feature"], importance_df["Importance"])
            ax.set_xlabel("Feature importance")
            ax.set_title("Random Forest feature importance")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(t["importance_note"])

        # Optional local SHAP explanation. If SHAP fails, keep the app working.
        st.subheader(t["shap_title"])
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_data)

            if isinstance(shap_values, list):
                local_values = np.asarray(shap_values[prediction_encoded]).reshape(-1)
            else:
                sv = np.asarray(shap_values)
                if sv.ndim == 3:
                    if sv.shape[0] == 1 and sv.shape[1] == len(features):
                        local_values = sv[0, :, prediction_encoded]
                    elif sv.shape[0] == 1 and sv.shape[2] == len(features):
                        local_values = sv[0, prediction_encoded, :]
                    else:
                        raise ValueError(f"Unexpected SHAP output shape: {sv.shape}")
                elif sv.ndim == 2:
                    local_values = sv[0]
                else:
                    raise ValueError(f"Unexpected SHAP output shape: {sv.shape}")

            shap_df = pd.DataFrame({
                "Feature": features,
                "SHAP value": np.asarray(local_values, dtype=float)
            }).sort_values("SHAP value")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(shap_df["Feature"], shap_df["SHAP value"])
            ax.axvline(0, linewidth=1)
            ax.set_xlabel("SHAP value")
            ax.set_title(f"SHAP explanation: {predicted_crop}")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(t["shap_note"])
        except Exception as shap_error:
            st.info(t["shap_unavailable"])
            with st.expander("Technical SHAP details"):
                st.code(str(shap_error))

        st.subheader(t["advisory"])
        st.info(t["advisory_text"])

    except Exception as exc:
        st.error(t["invalid"])
        st.exception(exc)
else:
    st.info(t["no_prediction"])

st.markdown("---")
st.caption(t["footer"])
st.caption(t["crop_names"])
