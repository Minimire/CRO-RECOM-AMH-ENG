# 🌾 CRO-RECOM-AMH-ENG: Bilingual Crop Recommendation System
### AI-Driven Precision Farming Technical Suite (Ethiopia Pilot Zone)
**Module 6: Final Deployment, Validation, Monitoring & Maintenance Framework**

| Field | Detail |
| :--- | :--- |
| **Course** | FUSE AI-201 — AI Model Development and Implementation |
| **Group Number** | 11 |
| **Module Project Stage** | Module 6 — Deploy, Validate & Maintain |
| **Submission Year** | November 2026 |
| **Primary Production Model** | Tuned Random Forest Classifier — 99.77% Test Accuracy |
| **Live Application URL** | **[INSERT YOUR STREAMLIT SHARE CLOUD URL HERE]** |

---

## 🔗 Continuity Statement
- **Module 4 Proposal (Approved 26 Sep 2026):** Formulated the 22-class problem statement, target users, data specifications, success metrics, and cloud deployment roadmap.
- **Module 5 Core Engine (Submitted October 2026):** Delivered data preparation pipelines, exploratory data visualizations (EDA), baseline metrics benchmarks, hyperparameter optimization tuning, SHAP explainability analyses, and a locally-hosted code prototype.
- **Module 6 Delivery (Current Stage):** Establishes public cloud serving infrastructure, introduces a native **Amharic-English Bilingual selection layer**, maps an **Ethiopian Agro-ecological Highland Zone warning matrix**, and details the strategic plans for data drift monitoring, automated retraining iterations, API extensions, and cold-start latency mitigation.

---

## 🌍 Project Overview & Regional Context
Agriculture serves as the foundation of Ethiopia's economy, contributing approximately **32% of national GDP**, sustaining **64% of the population**, and generating close to **79% of export earnings**. The sector is overwhelmingly driven by rural smallholder family farms cultivating roughly 95% of arable land. 

However, smallholder farmers generally lack access to site-specific soil testing frameworks due to high lab testing fees ($19–$23 per sample). Crop selection is consequently guided by traditional intuition or generalized, blanket national guidelines rather than site-specific soil chemistry. This data gap drives severe real-world penalties: regional wheat yields average 2.7 t/ha (nearly 50% below research station potentials), out-of-zone maize cultivation triggers steep yield penalties, and suboptimal input usage costs the national economy 295–946 million Birr annually.

This project delivers an end-to-end **22-class bilingual crop recommendation engine** that maps measured soil nutrients (N, P, K) and meteorological indicators (Temperature, Humidity, Soil pH, Rainfall) to the most appropriate crop with an attached inference confidence score.

---

## 🗺️ System Limitations & Implementation Constraints
While the predictive engine achieves an optimal mathematical accuracy profile of **99.77%** on the validation set, the team has identified distinct real-world boundary constraints that must be handled prior to scaling the pilot:

1. **Omission of Soil Physical Architecture:** The current input feature matrix maps chemical macronutrients (`N`, `P`, `K`) but completely omits physical soil variables such as soil texture profiles (clay, silt, sandy loam distribution), bulk density, compaction coefficients, and soil organic matter (SOM) thickness. These physical structural variables dictate real-world moisture retention bounds, root expansion capacity, and drainage physics.
2. **Topographic & Elevation Omissions:** The system is completely blind to regional topographical constraints. Slope percentage, landscape aspect, and micro-elevation terrain factors significantly control water runoff vectors, soil erosion variables, and local frost pockets across highly uneven valley terrains.
3. **Macro-Climate Static Averages vs. Short-Term Volatility:** Input sliders capture seasonal meteorological averages (e.g., historical rainfall and average temperature lines). In real-world operation, the engine is blind to immediate climate shocks, severe daily temperature fluctuations, unseasonal heatwaves, or flash flooding events during critical seed germination intervals.
4. **Economic & Supply Chain Disconnect:** The algorithmic output provides a technical agronomic suitability index but does not calculate micro-economic market variables, regional seed availability across local agricultural cooperatives, regional fertilizer distribution constraints, or localized market pricing volatility.

---

## 🇪🇹 Ethiopian Regional Localization Strategy
To address the primary geographical data gap (the underlying dataset represents general tropical and subtropical bounds), the codebase implements an interactive, multi-layered **Ethiopian Agro-ecological Highland Zone calibration check** within the user interface:

### 1. Agro-Ecological Matrix Mapping
The internal user system maps input parameters directly to established national altitudinal boundaries, flagging warning notices if a recommended crop type conflicts with typical regional parameters:

*   **Bereha Zone (Desert / Lowlands | < 500 meters):** Calibrated for high heat tolerance thresholds. Priority regional retraining focus includes Date Palm and extreme drought-tolerant Sorghum varieties.
*   **Kolla Zone (Warm Semi-Arid | 500 – 1,500 meters):** Characterized by high evaporation rates. Priority regional retraining focus includes Sorghum, Sesame, Cotton, and specific sub-types of Maize.
*   **Weina Dega Zone (Cool Sub-Humid Highlands | 1,500 – 2,300 meters):** Represents the primary target zone. Calibrated to evaluate Maize, Teff, Wheat varieties, and Haricot Beans.
*   **Dega Zone (Cool Alpine Highlands | 2,300 – 3,200 meters):** Calibrated for lower temperature bounds. Focuses tightly on Barley, cold-tolerant Wheat varieties, Faba Beans, and Potatoes.

### 2. Meteorological Seasonal Realignment
Instead of mapping absolute annual rainfall, the data pipeline is structurally designed to transition towards regional seasonal moisture patterns unique to East Africa:
-   **Kiremt Season:** Main rainy season (June–September), governing long-cycle crops.
-   **Belg Season:** Short rainy season (February–May), governing short-cycle, high-speed yields.
-   **Bega Season:** Dry season (October–January), critical for identifying crops suitable for residual moisture or minor irrigation diversion.

### 3. Local Data Source Integration Pathway
The upcoming deployment roadmap defines direct database ingestion pathways to retrain the core Random Forest model weights using authentic local agricultural repositories:
-   **EthioSIS (Ethiopian Soil Information System):** To pull localized macronutrient mapping vectors directly matching regional administrative zones (Woredas and Kebeleh).
-   **NMA (National Meteorology Agency):** To transition input variables from manual user sliders to localized historical gridded rainfall datasets.

---

## ⚙️ Cloud Infrastructure & Deployment Risk Mitigation
The architecture is deployed live onto the **Streamlit Community Cloud**. To guarantee runtime stability and completely bypass the resource constraints enforced by free cloud servers, the following engineering solutions were developed:

### 1. Memory Optimization (<1 GB RAM Cap Guard)
Live generation of **SHAP (Shapley Additive Explanations)** matrix weights requires computing game-theory distributions across all classes, which spikes memory usage and causes free servers to instantly crash. To prevent this, SHAP values were pre-rendered into static image assets (`shap_summary_global.png`) during training. The app renders this image asset instantly with **0 MB of processing memory overhead**.

### 2. Eliminating Python Repository Bloat
Heavy analytical development packages (such as `optuna` and `shap`) were stripped out of the production `requirements.txt` file. This reduces container building size, eliminates dependency conflict risks, and optimizes real-time user slider response latencies to **under 10 milliseconds**.

### 3. Pipeline Mismatch Protection
The standard scalar properties, categorical target label encoders, and optimized Random Forest model parameters were serialized into a single compound dictionary file (`crop_recommendation_system_bundle.pkl`) to guarantee zero-mismatch errors between local preprocessing states and live cloud inference runs.

---

## ⚡ Mitigating the Container "Cold Start" Problem
Free hosting platforms put applications into a "hibernation" or "sleep" state after periods of user inactivity to conserve server resources. The next user accessing the app faces a long, notable latency delay (a **Cold Start**) while the server container pulls your files, installs packages, and re-boots the framework. 

To mitigate and bypass this cold start delay, our system implements three defensive architectural solutions:

1. **Bypassing Heavy Packages:** By removing `optuna` and `shap` from `requirements.txt`, the installation phase during a container rebuild drops from a painful 3–4 minutes down to **under 20 seconds**.
2. **Resource Caching Logic:** The main model bundle is wrapped inside a `@st.cache_resource` function decorator. This forces the server to read and store model weights directly into the container's virtual memory cache *only once* upon initialization, ensuring subsequent user slider adjustments process instantly with zero repeated processing delays.
3. **External Keep-Alive Cron Pings (Production Plan):** For formal field pilot deployment, an automated cloud uptime script (e.g., UptimeRobot or an external GitHub Action cron-job) is configured to send a lightweight automated web ping to your Streamlit URL every 14 minutes. This prevents the server from entering sleep mode, guaranteeing 100% immediate availability for agricultural field extension workers.

---


 API Integration Recommendations

To scale this standalone web prototype into a distributed, production-grade network across regional Ethiopian cooperative networks, the following REST API Architecture is recommended:
• Backend Serving Layer: Decouple the user interface from the model completely by packaging the serialized crop_recommendation_system_bundle.pkl file inside a FastAPI REST API container wrapped inside a Docker image.
• Automated Meteorological Endpoint: Build automated data retrieval endpoints that take incoming user GPS coordinates and dynamically pull live temperature, relative humidity, and precipitation averages from the Open-Meteo API or the National Meteorology Agency (NMA) grid. This eliminates human slider errors by populating weather parameters automatically.
• Inference Schema Structure: The API endpoint will accept standard POST JSON payloads and return a structured response payload:
json { "gps_coordinates": {"latitude": 9.02, "longitude": 38.74}, "predicted_crop": "wheat", "amharic_label": "ስንዴ", "confidence_score": 0.9823, "agro_ecological_zone": "Weina Dega" } 

📈 Future Roadmap Enhancements

To address current system data gaps and build towards a highly localized, adaptive agricultural advisory network, the following future roadmap steps are prioritized:
1. Ingesting Localized Diagnostic Soil Surveys: Replace general South Asian baseline conditions by directly retraining model branches on regional African datasets sourced from EthioSIS (Ethiopian Soil Information System).
2. SMS/USSD Mobile Framework Integration: Because internet connectivity is highly constrained across remote rural Ethiopian smallholder farm boundaries, the FastAPI microservice will be linked to an SMS/USSD gateway, allowing farmers with basic analog phones to text code vectors (e.g., *812#) and receive crop predictions via offline SMS texts.
3. Macro-Economic Market Viability Filter: Connect agronomic suitability predictions with real-time seasonal crop market price datasets from the Ethiopian Commodity Exchange (ECX). This ensures the engine doesn't just recommend a crop because it grows well, but because it is economically profitable for the family to cultivate.

👥 Team Contributions & Operational Credits

• Domain & Problem Research: Mihret Endeshaw, Biniyam Meldeye, Eyuel Getachew (Agro-ecological profiling and local impact metrics definition).
• Data Engineering & Preprocessing: Samson Amdissa, Yonas Debolo, Mihret Endeshaw (Data hygiene audits, stratified split partitioning, pipeline serialization).
• Machine Learning Engineering: Eyuel Getachew, Fikerachen Sisay, Yonas Debolo (GridSearchCV tuning, Optuna trial optimization, multi-baseline matrix scoring, SVM integration).
• Software Architecture & Deployment: Sisay Yeshak, Milkoma Ayele, Temesgen Teshome (Amharic translation dictionary design, Streamlit bundle integration, cloud hosting setup).
• Quality Assurance & Verification: Temesgen Teshome, Boniyas Wagari, Samson Amdissa (Boundary validation checks, drought scenario stress testing, error matrix tracking).

📚 18. References (APA Style)

Aingle, A. (2020). Crop recommendation dataset [Data set]. Kaggle. kaggle.com
Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794). ACM.
Geetha, V., Punitha, A., Abarna, S., & Janani, P. (2020). An effective crop prediction using random forest algorithm. International Journal of Advanced Science and Technology, 29(6), 2367–2374.
Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30, 4765–4774.
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
Tharun, R., Babu, R., & Mohebbanaaz, S. (2026). Comparative analysis of RF, SVM and XGBoost for high-accuracy crop recommendation. In 2026 IEEE Madhya Pradesh Section Conference (MPCON). IEEE.
Wale, E. (2003). What do farmers financially lose if they fail to use improved seeds? Some econometric results for wheat and implications for agricultural extension policy in Ethiopia. University of Minnesota.
