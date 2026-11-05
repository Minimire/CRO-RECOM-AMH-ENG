# 🌾 CRO-RECOM-AMH-ENG: Bilingual Crop Recommendation System
### AI-Driven Precision Farming Technical Suite (Ethiopia Pilot)
**Module 6: Deployment, Validation, Monitoring & Maintenance Report**

| Field | Detail |
| :--- | :--- |
| **Course** | FUSE AI-201 — AI Model Development and Implementation |
| **Group Number** | 11 |
| **Module Project Stage** | Module 6 — Deploy, Validate & Maintain |
| **Submission Date** | November 2026 |
| **Primary Model** | Tuned Random Forest Classifier — 99.77% Test Accuracy |
| **Live Application** | **[INSERT YOUR STREAMLIT SHARE CLOUD URL HERE]** |

---

## 🔗 Continuity Statement
- **Module 4 Proposal (Approved 26 Sep 2026):** Defined the multi-class problem, target users, dataset, success criteria, and cloud deployment roadmap.
- **Module 5 Core Engine (Submitted October 2026):** Implemented data preparation scripts, EDA, baseline benchmarks, hyperparameter optimization tuning, SHAP explainability matrices, and a locally hosted Streamlit prototype.
- **Module 6 Delivery (Current Stage):** Achieves production cloud deployment, introduces an **Amharic-English Bilingual interface**, integrates an **Ethiopian Agro-ecological Highland Zone warning matrix**, and designs structured system monitoring/maintenance scripts for long-term operational survival.

---

## 🌍 Project Overview & Regional Context
Agriculture is the foundation of Ethiopia's economy, contributing roughly **32% of national GDP**, sustaining **64% of the population**, and generating close to **79% of export earnings**. The sector is overwhelmingly driven by rural smallholder family farms cultivating 95% of arable land. 

However, smallholder farmers generally lack access to site-specific soil testing frameworks due to high lab fees ($19–$23 per sample). Crop selection is consequently guided by traditional intuition or generalized, blanket national fertilizer recommendations. This data gap drives severe real-world penalties: regional wheat yields average 2.7 t/ha (nearly 50% below research station potentials), out-of-zone maize cultivation triggers steep yield penalties, and suboptimal agronomic input usage costs the national economy 295–946 million Birr annually.

This project delivers an end-to-end **22-class bilingual crop recommendation engine** that maps measured soil nutrients (N, P, K) and meteorological indicators (Temperature, Humidity, Soil pH, Rainfall) to the most appropriate crop with an attached inference confidence score.

> ⚠️ **Pilot Status Advisory:** The underlying training dataset represents general tropical/subtropical boundaries (Kaggle Dataset). This system is positioned as a validated pilot and requires local calibration before unconstrained field deployment across Ethiopian highland regimes.

---

## 🗺️ Localization Strategy (Ethiopian Context)
To align the South Asian database profiles with real-world Ethiopian farming needs, the deployment codebase introduces a manual **Agro-Ecological Highland Zone calibration check** directly in the interface:

| Agro-Ecological Zone | Elevation Bracket | Priority Crops for Retraining / Cross-Reference |
| :--- | :--- | :--- |
| **Bereha** | < 500 m | Date Palm, drought-tolerant Sorghum |
| **Kolla** | 500 – 1,500 m | Sorghum, Sesame, Cotton, Maize |
| **Weina Dega** | 1,500 – 2,300 m | Maize, Teff, Wheat, Haricot Bean |
| **Dega** | 2,300 – 3,200 m | Barley, Wheat, Faba Bean, Potato |

### Retraining & Calibration Pathway
- **Local Data Ingestion:** Future pipelines will ingest localized regional profiles from **EthioSIS** (Ethiopian Soil Information System) and the **National Meteorology Agency (NMA)**.
- **Seasonal Realignment:** Metric matrices will map directly to local moisture regimes—**Kiremt** (June–September), **Belg** (February–May), and **Bega** (October–January)—rather than absolute annual averages.
- **Access Architecture:** Features native **Amharic (አማርኛ)** and English translation selection layers alongside an expansion architecture designed for lightweight Android/USSD offline deployments.

---

## 📊 Model Development, Baselining, and Evaluation Matrix
The core engine was evaluated using a stratified 80/20 train-test validation partition to preserve strict minority boundary thresholds. In accordance with Module 4 requirements, **Support Vector Machines (SVM)** were explicitly introduced to reinforce our baseline evaluation matrix:

| Model Classification | Algorithm Suite | Validation Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Advanced Ensemble** | **Random Forest (Tuned Champion)** | **99.77%** | **99.80%** | **99.77%** | **99.77%** |
| Advanced Ensemble | XGBoost (Optimized Challenger) | 99.77% | 99.79% | 99.77% | 99.77% |
| Classical Baseline | Support Vector Machine (SVM) | 97.95% | 98.05% | 97.95% | 97.93% |
| Classical Baseline | K-Nearest Neighbors (KNN, k=5) | 97.73% | 97.91% | 97.73% | 97.71% |
| Classical Baseline | Logistic Regression | 96.82% | 97.12% | 96.82% | 96.81% |
| Control Baseline | Dummy Classifier (Most Frequent) | 3.41% | 0.12% | 3.41% | 0.22% |

### Scope & Tuning Reconciliation
- **Random Forest Optimization:** Fine-tuned via **GridSearchCV** over parameters mapping tree depths and minimum sample splits (`max_depth=[8, 12, None]`, `min_samples_split=[2, 5]`), converging at a peak accuracy profile of 99.77%.
- **XGBoost Optimization:** Fine-tuned over 15 cross-validated **Optuna** search trials to minimize logarithmic loss while strictly monitoring container compute overhead.
- **Baseline Performance Analysis:** The high baseline scores of SVM (97.95%) and KNN (97.73%) demonstrate that agronomic profiles are highly linearly separable in nature, clustering cleanly with minimal regional noise.

---

## ⚙️ Cloud Infrastructure & Deployment Risk Mitigation
The architecture is deployed to the **Streamlit Community Cloud** with standard container serving profiles. To guarantee stability and bypass the free hosting tier constraints, the following architectural choices were made:

1. **RAM Footprint Restriction (<1 GB hard cap):** Live execution of **SHAP (Shapley Additive Explanations)** models spikes memory allocation. To prevent container crashes, SHAP values were pre-rendered into static image assets (`shap_summary_global.png`) during training. The application renders this asset file instantly with 0% container memory overhead.
2. **Cold-Start Latency Protection:** Heavy analytical packages (like `optuna` and `shap`) were completely omitted from the production `requirements.txt`. This simplifies the image compilation loop, prevents repository bloat, and optimizes user response times (<10 ms).
3. **Pipeline Artifact Bundling:** The scalar parameters, target label encoders, and optimized Random Forest model parameters were serialized into a single compound dictionary file (`crop_recommendation_system_bundle.pkl`) to guarantee zero-mismatch errors between local development and cloud runtimes.

---

## 🛠️ Post-Deployment Monitoring & Maintenance Framework

To ensure the longevity of the solution, the system follows a structured **Data Drift and Model Degradation** monitoring cycle:

### 1. Soil and Climate Data Drift Tracking
As climate patterns shift in East Africa, incoming feature ranges will be logged and evaluated against the baseline training distribution using **Population Stability Index (PSI)** scoring:
- **PSI < 0.1:** Distribution is stable; no system action required.
- **0.1 ≤ PSI < 0.25:** Moderate drift detected; triggers system logs for administrative review.
- **PSI ≥ 0.25:** Severe environmental drift; flags the model pipeline for automated retraining.

### 2. Scheduled Retraining Iterations
- **Telemetry Log Audits:** Input data shapes from extension workers will be batched into temporary, anonymous cloud data buckets monthly.
- **Retraining Trigger:** The system is scheduled for formal performance audits semi-annually. If performance dips below the **Module 4 primary metric threshold (Macro F1-Score < 0.90)** due to environmental changes, the model will be retrained with the new data points.

### 3. Graceful Failure Safeguard
If an operator inputs a parameter combination that falls outside standard arable ranges, or if the model confidence falls below **50% certainty**, the Streamlit UI suppresses automated crop outputs and provides a clear **Low Confidence Support Alert** instructing the user to double-check entries and consult local extension agronomists.

---

## 👥 Team Contributions & Operational Credits
- **Domain & Problem Research:** Mihret Endeshaw, Biniyam Meldeye, Eyuel Getachew (Agro-ecological profiling and local impact metrics definition).
- **Data Engineering & Preprocessing:** Samson Amdissa, Yonas Debolo, Mihret Endeshaw (Data hygiene audits, stratified split partitioning, pipeline serialization).
- **Machine Learning Engineering:** Eyuel Getachew, Fikerachen Sisay, Yonas Debolo (GridSearchCV tuning, Optuna trial optimization, multi-baseline matrix scoring, SVM integration).
- **Software Architecture & Deployment:** Sisay Yeshak, Milkoma Ayele, Temesgen Teshome (Amharic translation dictionary design, Streamlit bundle integration, cloud hosting setup).
- **Quality Assurance & Verification:** Temesgen Teshome, Boniyas Wagari, Samson Amdissa (Boundary validation checks, drought scenario stress testing, error matrix tracking).
