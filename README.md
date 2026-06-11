# 🫀 Heart Disease Prediction Web Application

A full-stack machine learning web application that predicts heart disease risk from 14 clinical features using five trained ML models, with personalized health recommendations in English and Khmer.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset & Features](#dataset--features)
- [Machine Learning Models](#machine-learning-models)
- [Model Performance](#model-performance)
- [API Reference](#api-reference)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)

---

## Overview

This project predicts heart disease risk by accepting 14 clinical input features and running them through one of five trained classification models. The backend API returns:

- A **binary prediction** (heart disease / healthy)
- A **risk probability score** (0–100%)
- A **risk level** classification (Low / Moderate / High)
- A **verdict** statement
- **Personalized health recommendations** tailored to the patient's clinical values

The frontend provides two pages:
- **Landing page** (`index.html`) — project overview and information
- **Prediction page** (`prediction-page.html`) — interactive form with real-time results, risk gauge, and voice narration (TTS)

Bilingual support: **English** and **Khmer (ភាសាខ្មែរ)**.

---

## Project Structure

```
NewWeb/
├── frontend/
│   ├── index.html              # Landing / home page
│   ├── prediction-page.html    # Interactive prediction form & results
│   ├── css/
│   │   └── styles.css
│   ├── fonts/
│   └── images/
│       └── clinic.png
│
├── api/
│   ├── app.py                  # FastAPI backend (5 models, /predict, /tts endpoints)
│   ├── requirements.txt        # Python dependencies
│   └── model/
│       ├── Final_code.ipynb                        # Full ML training notebook
│       ├── heart_disease_data_with_features.csv    # Raw dataset with engineered features
│       ├── cleaned_heart_disease_data.csv          # Cleaned dataset
│       ├── logistic_regression_model.pkl
│       ├── random_forest_model.pkl
│       ├── svm_model.pkl
│       ├── xgboost_model.pkl
│       └── lightgbm_model.pkl
│
├── howtorun.txt
└── README.md
```

---

## Dataset & Features

### Source
The dataset (`heart_disease_data_with_features.csv`) is based on the classic UCI Heart Disease dataset. After cleaning and feature engineering, 14 final features are used for training.

**Train / Test Split:** 80% training / 20% testing (stratified, `random_state=42`)

### Input Features

| # | Feature | Description | Type / Range |
|---|---------|-------------|--------------|
| 1 | `Age` | Patient age | int, 20–100 |
| 2 | `Gender` | Gender (1 = male, 0 = female) | binary |
| 3 | `Chest_Pain_Type` | 1: Typical angina, 2: Atypical angina, 3: Non-anginal pain, 4: Asymptomatic | int, 1–4 |
| 4 | `Resting_Blood_Pressure` | Resting BP in mmHg | float, 80–220 |
| 5 | `Cholesterol` | Serum cholesterol in mg/dL | float, 100–600 |
| 6 | `Fasting_Blood_Sugar` | Fasting blood sugar > 120 mg/dL (1 = true) | binary |
| 7 | `Resting_ECG_Results` | 0: Normal, 1: ST-T abnormality, 2: LV hypertrophy | int, 0–2 |
| 8 | `Maximum_Heart_Rate` | Max heart rate achieved during exercise | float, 60–220 |
| 9 | `Exercise_Induced_Angina` | Exercise-induced chest pain (1 = yes) | binary |
| 10 | `Depression_Induced_By_Exercise` | ST depression induced by exercise (oldpeak) | float, 0–10 |
| 11 | `Slope_Of_Peak_Exercise` | 1: Upsloping, 2: Flat, 3: Downsloping | int, 1–3 |
| 12 | `Major_Vessels_Colored_By_Fluoroscopy` | Number of major vessels colored (0–3) | int, 0–3 |
| 13 | `Thalassemia` | 3: Normal, 6: Fixed defect, 7: Reversible defect | int, 3–7 |
| 14 | `Risk_Score` | **Engineered feature:** `(Age × Cholesterol / 1000) + (Resting_BP / 100)` | float |

### Data Preprocessing
- **Missing values:** Numerical columns → median imputation; Categorical → mode imputation
- **Duplicates:** Checked and removed
- **Scaling:** Applied only for Logistic Regression and SVM (`StandardScaler`)
- **Tree-based models** (Random Forest, XGBoost, LightGBM) use raw unscaled features

---

## Machine Learning Models

Five models were trained and evaluated on the same held-out test set:

### 1. Logistic Regression
- **Type:** Linear classifier
- **Preprocessing:** StandardScaler required
- **Parameters:** `max_iter=1000`, `random_state=42`
- **Feature Importance:** Coefficient magnitudes
- **Best for:** Interpretability, clinical baseline

### 2. Random Forest
- **Type:** Bagging ensemble (100 decision trees)
- **Preprocessing:** No scaling required
- **Parameters:** `n_estimators=100`, `max_depth=None`, `random_state=42`
- **Feature Importance:** Gini impurity
- **Best for:** Robust non-linear classification, handles outliers well

### 3. SVM (RBF Kernel)
- **Type:** Kernel method
- **Preprocessing:** StandardScaler required
- **Feature Importance:** Permutation importance (no linear coefficients)
- **Best for:** Small datasets, non-linear decision boundaries

### 4. XGBoost
- **Type:** Gradient boosting ensemble
- **Preprocessing:** No scaling required
- **Parameters:** `n_estimators=100`, `max_depth=4`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`
- **Feature Importance:** Gain / weight
- **Best for:** High accuracy on tabular data

### 5. LightGBM
- **Type:** Leaf-wise gradient boosting
- **Preprocessing:** No scaling required
- **Parameters:** `n_estimators=100`, `max_depth=4`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`
- **Feature Importance:** Gain / weight
- **Best for:** Fast training, large datasets, best ROC-AUC

---

## Model Performance

All models evaluated on the same 20% held-out test set (61 samples):

| Model | Test Accuracy | ROC-AUC | Feature Scaling | Type |
|-------|:-------------:|:-------:|:---------------:|------|
| **Random Forest** 🏆 | **91.80%** | 0.9410 | Not required | Bagging Ensemble |
| Logistic Regression | 86.89% | 0.9513 | Required | Linear |
| SVM (RBF) | 86.89% | 0.9459 | Required | Kernel Method |
| XGBoost | 85.25% | 0.9448 | Not required | Boosting Ensemble |
| **LightGBM** | 83.61% | **0.9524** 🏆 | Not required | Boosting Ensemble |

> **Random Forest** achieves the highest accuracy (91.80%).  
> **LightGBM** achieves the highest ROC-AUC (0.9524).

### Key Feature Importance Findings
- `Major_Vessels_Colored_By_Fluoroscopy` — strongest predictor across all models
- `Chest_Pain_Type` — second most important feature
- `Depression_Induced_By_Exercise` (ST depression) — strong cardiac stress indicator
- `Maximum_Heart_Rate` — negatively correlated (higher is healthier)
- `Thalassemia` — blood flow test result is highly informative
- `Risk_Score` (engineered) — contributes additional predictive signal

---

## API Reference

The backend is a **FastAPI** application running at `http://localhost:8000`.

### `GET /`
Health check — returns loaded model names.

```json
{ "status": "ok", "models_loaded": ["logistic", "rf", "svm", "xgboost", "lightgbm"] }
```

---

### `POST /predict`

Predicts heart disease risk for a patient.

**Request body:**

```json
{
  "age": 55,
  "gender": 1,
  "chest_pain_type": 4,
  "resting_bp": 140,
  "cholesterol": 250,
  "fasting_bs": 0,
  "resting_ecg": 1,
  "max_heart_rate": 115,
  "exercise_angina": 1,
  "oldpeak": 2.3,
  "st_slope": 2,
  "ca": 1,
  "thal": 7,
  "model": "rf",
  "lang": "en"
}
```

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `age` | float | 20–100 | Patient age |
| `gender` | int | 0–1 | 0 = female, 1 = male |
| `chest_pain_type` | int | 1–4 | Chest pain type |
| `resting_bp` | float | 80–220 | Resting blood pressure (mmHg) |
| `cholesterol` | float | 100–600 | Serum cholesterol (mg/dL) |
| `fasting_bs` | int | 0–1 | Fasting blood sugar > 120 mg/dL |
| `resting_ecg` | int | 0–2 | Resting ECG result |
| `max_heart_rate` | float | 60–220 | Maximum heart rate |
| `exercise_angina` | int | 0–1 | Exercise-induced angina |
| `oldpeak` | float | 0–10 | ST depression (exercise vs rest) |
| `st_slope` | int | 1–3 | Slope of peak exercise ST segment |
| `ca` | int | 0–3 | Number of major vessels (fluoroscopy) |
| `thal` | int | 3–7 | Thalassemia type |
| `model` | string | — | `"logistic"`, `"rf"`, `"svm"`, `"xgboost"`, `"lightgbm"` |
| `lang` | string | — | `"en"` (English) or `"km"` (Khmer) |

**Response:**

```json
{
  "model_used": "rf",
  "risk_score": 0.7823,
  "risk_percent": 78,
  "prediction": 1,
  "risk_level": "High",
  "verdict": "Elevated Risk — Clinical evaluation is strongly recommended.",
  "recommendations": [
    "🏥 Consult a cardiologist promptly...",
    "💊 Manage high blood pressure...",
    "..."
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `model_used` | string | Which model was used |
| `risk_score` | float | Raw probability (0.0–1.0) |
| `risk_percent` | int | Risk percentage (0–100) |
| `prediction` | int | 0 = Healthy, 1 = Heart Disease |
| `risk_level` | string | Low / Moderate / High |
| `verdict` | string | Human-readable verdict |
| `recommendations` | list | Personalized health tips |

**Risk thresholds:**
- `risk_score < 0.35` → **Low** risk
- `0.35 ≤ risk_score < 0.60` → **Moderate** risk
- `risk_score ≥ 0.60` → **High** risk

---

### `POST /tts`

Converts text to speech (MP3 audio) using Google TTS.

```json
{ "text": "Your heart disease risk is low.", "lang": "en" }
```

Returns `audio/mpeg` binary data.

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- pip

### 1. Clone / navigate to the project

```bash
cd api/
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
pydantic>=2.0.0
scikit-learn>=1.4.0
numpy>=1.26.0
joblib>=1.4.0
xgboost>=2.0.0
lightgbm>=4.3.0
```

> **Note:** The `gTTS` package is also required for the TTS endpoint. Install it separately:
> ```bash
> pip install gTTS
> ```

### 3. Ensure model files are present

The `api/model/` directory must contain all five `.pkl` model files:
- `logistic_regression_model.pkl`
- `random_forest_model.pkl`
- `svm_model.pkl`
- `xgboost_model.pkl`
- `lightgbm_model.pkl`

If they are missing, run `Final_code.ipynb` end-to-end to retrain and save all models.

---

## How to Run

### Start the API server

From the `api/` directory:

```bash
uvicorn app:app --reload
```

The API will be available at: **`http://localhost:8000`**

Interactive API docs (Swagger UI): **`http://localhost:8000/docs`**

### Open the frontend

Open `frontend/index.html` directly in a browser, or serve it with any static file server.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Backend API** | Python, FastAPI, Uvicorn |
| **ML Models** | scikit-learn, XGBoost, LightGBM |
| **Data Processing** | pandas, NumPy |
| **Model Persistence** | joblib, pickle |
| **Text-to-Speech** | gTTS (Google Text-to-Speech) |
| **Notebook** | Jupyter Notebook (`Final_code.ipynb`) |

---

## Notebook Structure (`Final_code.ipynb`)

The training notebook is organized into 9 parts:

| Part | Title |
|------|-------|
| 1 | Data Loading & Cleaning |
| 2 | Exploratory Data Analysis (EDA) |
| 3 | Feature Engineering & Train/Test Split |
| 4 | Logistic Regression |
| 5 | Random Forest |
| 6 | Support Vector Machine (SVM) |
| 7 | Gradient Boosting (XGBoost & LightGBM) |
| 8 | Model Comparison |
| 9 | Save All Models |

---

> ⚠️ **Disclaimer:** This application is intended for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.