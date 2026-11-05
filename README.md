# Intelligent Crop Recommendation System

An AI-driven agricultural decision-support microservice pipeline designed to recommend optimal crops based on environmental parameters. The system features multilingual support (English and Amharic), hyperparameter optimization via Optuna, model explainability using SHAP, and a dual-service deployment architecture utilizing FastAPI and Streamlit.

## ðŸš€ Live Application
The interactive dashboard is deployed and accessible live:
ðŸ‘‰ **[Live Streamlit Dashboard](https://cro-recom-amh-eng-mih.streamlit.app/#model-interpretability-and-explainable-ai-xai)**

---

## ðŸ“Œ Project Architecture
The system is built as a modular pipeline distributed across specialized microservices:
1. **Data Ingestion & Preprocessing:** Cleans and processes agricultural tabular datasets (sourced from KaggleHub).
2. **Exploratory Data Analysis (EDA):** Visualizes environmental distributions and feature correlations.
3. **Model Training & Optimization:** Employs Optuna for hyperparameter tuning.
4. **FastAPI Backend:** Serves the core ML models via a REST API microservice.
5. **Streamlit UI:** Provides an intuitive, multilingual dashboard featuring live inference, baseline evaluations, and SHAP explainability visualizations.

---

## ðŸ“Š Model Pipeline & Baseline Evaluation
The system evaluates a comprehensive suite of classifiers to ensure optimal deployment accuracy:

*   **Primary Model:** Random Forest Classifier (Optimized via Optuna)
*   **Challenger Model:** XGBoost
*   **Baseline Options:** Logistic Regression, K-Nearest Neighbors (KNN), and Dummy Classifier

### Performance Summary
The models are evaluated across standard classification metrics:
*   **Accuracy / Precision / Recall / F1-Score**
*   **Confusion Matrix Interpretation:** Evaluates specific class-error tendencies to ensure high-fidelity agricultural advice before real-world deployment.

---

## ðŸ” Model Interpretability & Explainable AI (XAI)
To build stakeholder trust, the application integrates **SHAP (SHapley Additive exPlanations)**. This allows farmers and agricultural specialists to:
*   Understand individual feature contributions (e.g., Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall) per recommendation.
*   Visualize global feature importance to interpret overall model behavior.

---

## ðŸ› ï¸ Tech Stack
*   **Development Core:** Google Colab, Python 3.x
*   **Machine Learning Frameworks:** Scikit-Learn, XGBoost, Optuna, SHAP
*   **API Layer:** FastAPI, Uvicorn
*   **Dashboard Layer:** Streamlit (supporting English/Amharic UI selection)
*   **Containerization:** Docker

---

## âš™ï¸ Direct Deployment & Setup

### 1. Backend Microservice (FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 2. Frontend Dashboard (Streamlit)
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

### 3. Containerized Deployment (Docker Compose)
```bash
docker-compose up --build
```

---

## ðŸ”® Future Improvements
*   Integration of deep learning architectures (e.g., Recurrent Neural Networks for time-series climate forecasting).
*   Expansion of local language support and real-time IoT soil sensor integration.
