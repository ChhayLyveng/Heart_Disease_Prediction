"""
Heart Disease Prediction API
FastAPI backend — loads all 5 trained ML models from the model/ folder
and exposes a /predict endpoint used by the frontend prediction page.
"""

import joblib
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, root_validator
from fastapi.responses import Response
from gtts import gTTS
import io

# ─── Translations Dictionary ──────────────────────────────────────────────────
TRANSLATIONS = {
    "en": {
        "verdict_lower": "Lower Risk — Heart disease is unlikely based on your profile.",
        "verdict_moderate": "Moderate Risk — Some indicators warrant further monitoring.",
        "verdict_high": "Elevated Risk — Clinical evaluation is strongly recommended.",
        "risk_level_low": "Low",
        "risk_level_moderate": "Moderate",
        "risk_level_high": "High",
        
        "rec_high_risk": "🏥 **Consult a cardiologist promptly** — your profile indicates elevated risk of heart disease.",
        "rec_low_risk": "✅ **Maintain your healthy lifestyle** — your current risk level is lower, but regular monitoring is key.",
        "bp_high": "💊 **Manage high blood pressure** — your resting BP ({:.0f} mmHg) is above the normal threshold (120 mmHg). Speak with your doctor about medication or lifestyle changes.",
        "bp_elevated": "⚠️ **Monitor blood pressure** — your resting BP ({:.0f} mmHg) is elevated. Reduce salt intake and increase physical activity.",
        "chol_high": "🥗 **Reduce cholesterol** — your serum cholesterol ({:.0f} mg/dL) is high. Adopt a low-fat, high-fibre diet and consider statins after discussing with your doctor.",
        "chol_borderline": "🥗 **Watch your diet** — your cholesterol ({:.0f} mg/dL) is borderline. Favour fruits, vegetables, and omega-3-rich foods.",
        "bs_high": "🍬 **Control blood sugar** — fasting blood sugar >120 mg/dL is a significant risk factor. Limit refined sugars and consult a specialist about diabetes management.",
        "hr_low": "🏃 **Improve cardiovascular fitness** — your maximum heart rate ({:.0f} bpm) is low. Gradual aerobic exercise (walking, cycling) can strengthen the heart.",
        "angina": "⛔ **Avoid strenuous exertion until assessed** — exercise-induced chest pain (angina) requires clinical evaluation before continuing vigorous activity.",
        "st_dep": "📈 **ST depression is elevated** ({:.1f} mm) — this can indicate myocardial ischemia. An ECG stress test with a cardiologist is strongly advised.",
        "cp_asymp": "🫀 **Asymptomatic chest pain pattern** — silent ischemia is harder to detect. Proactive cardiac screening (ECG, echo) is recommended.",
        "cp_typ": "⚠️ **Typical angina detected** — schedule a cardiology review to assess coronary artery disease risk.",
        "vessels_multi": "🔬 **Multiple blocked vessels noted** ({} vessels) — this is a strong indicator of coronary artery disease. Immediate specialist review is essential.",
        "vessels_one": "🔬 **One blocked vessel** — monitor closely and discuss angiography options with your cardiologist.",
        "thal_rev": "🩺 **Reversible defect in blood flow test** — this suggests stress-induced ischemia. A nuclear stress test review is warranted.",
        "thal_fixed": "🩺 **Fixed defect in blood flow test** — may indicate prior infarction. Discuss with your doctor.",
        "age_risk": "📅 **Annual cardiac check-ups are essential** — risk increases with age. Ensure regular ECG, lipid panel, and blood pressure monitoring.",
        "smoke": "🚭 **Avoid smoking** — tobacco is the leading modifiable cardiovascular risk factor.",
        "sleep": "😴 **Prioritise sleep** — aim for 7–8 hours per night; poor sleep elevates heart disease risk."
    },
    "km": {
        "verdict_lower": "ហានិភ័យទាប — ជំងឺបេះដូងមិនទំនងមានទេ ដោយផ្អែកលើទិន្នន័យរបស់អ្នក។",
        "verdict_moderate": "ហានិភ័យមធ្យម — សូចនាករមួយចំនួនតម្រូវឱ្យមានការតាមដានបន្ថែមទៀត។",
        "verdict_high": "ហានិភ័យខ្ពស់ — ការវាយតម្លៃតាមគ្លីនិកត្រូវបានណែនាំយ៉ាងខ្លាំង។",
        "risk_level_low": "ទាប",
        "risk_level_moderate": "មធ្យម",
        "risk_level_high": "ខ្ពស់",
        
        "rec_high_risk": "🏥 **សូមពិគ្រោះជាមួយគ្រូពេទ្យឯកទេសបេះដូងជាបន្ទាន់** — ទិន្នន័យរបស់អ្នកបង្ហាញពីហានិភ័យខ្ពស់នៃជំងឺបេះដូង។",
        "rec_low_risk": "✅ **រក្សារបៀបរស់នៅប្រកបដោយសុខភាពល្អរបស់អ្នក** — កម្រិតហានិភ័យបច្ចុប្បន្នរបស់អ្នកគឺទាប ប៉ុន្តែការតាមដានជាប្រចាំគឺសំខាន់។",
        "bp_high": "💊 **គ្រប់គ្រងសម្ពាធឈាមខ្ពស់** — សម្ពាធឈាមសម្រាករបស់អ្នក ({:.0f} mmHg) គឺលើសកម្រិតធម្មតា (120 mmHg)។ សូមពិភាក្សាជាមួយគ្រូពេទ្យរបស់អ្នកអំពីថ្នាំ ឬការផ្លាស់ប្តូររបៀបរស់នៅ។",
        "bp_elevated": "⚠️ **តាមដានសម្ពាធឈាម** — សម្ពាធឈាមសម្រាករបស់អ្នក ({:.0f} mmHg) កំពុងកើនឡើង។ កាត់បន្ថយការទទួលទានអំបិល និងបង្កើនសកម្មភាពរាងកាយ។",
        "chol_high": "🥗 **កាត់បន្ថយកូឡេស្តេរ៉ុល** — កូឡេស្តេរ៉ុលក្នុងឈាមរបស់អ្នក ({:.0f} mg/dL) គឺខ្ពស់។ ទទួលទានអាហារមានជាតិខ្លាញ់ទាប និងជាតិសរសៃខ្ពស់ ហើយពិចារណាប្រើថ្នាំ statins បន្ទាប់ពីពិភាក្សាជាមួយគ្រូពេទ្យ។",
        "chol_borderline": "🥗 **ពិនិត្យរបបអាហាររបស់អ្នក** — កូឡេស្តេរ៉ុលរបស់អ្នក ({:.0f} mg/dL) គឺស្ថិតនៅបន្ទាត់ព្រំដែន។ គួរទទួលទានផ្លែឈើ បន្លែ និងអាហារសម្បូរអូមេហ្គា 3។",
        "bs_high": "🍬 **គ្រប់គ្រងជាតិស្ករក្នុងឈាម** — ជាតិស្ករពេលតមអាហារ >120 mg/dL គឺជាកត្តាហានិភ័យសំខាន់។ កម្រិតជាតិស្ករចម្រាញ់ និងពិគ្រោះជាមួយអ្នកឯកទេសអំពីការគ្រប់គ្រងជំងឺទឹកនោមផ្អែម Ly។",
        "hr_low": "🏃 **ធ្វើឱ្យប្រសើរឡើងនូវកាយសម្បទាបេះដូងសរសៃឈាម** — អត្រាចង្វាក់បេះដូងអតិបរមារបស់អ្នក ({:.0f} bpm) គឺទាប។ លំហាត់ប្រាណតាមបែបអេរ៉ូប៊ិកបន្តិចម្តងៗ (ការដើរ ជិះកង់) អាចពង្រឹងបេះដូង។",
        "angina": "⛔ **ជៀសវាងការប្រឹងប្រែងខ្លាំងរហូតដល់មានការវាយតម្លៃ** — ការឈឺទ្រូងដោយសារលំហាត់ប្រាណ (angina) ទាមទារការវាយតម្លៃគ្លីនិកមុនពេលបន្តសកម្មភាពខ្លាំងក្លា។",
        "st_dep": "📈 **ការធ្លាក់ចុះ ST គឺខ្ពស់** ({:.1f} mm) — នេះអាចបង្ហាញពីបញ្ហាកង្វះឈាមទៅចិញ្ចឹមសាច់ដុំបេះដូង។ ការធ្វើតេស្ត ECG ជាមួយគ្រូពេទ្យបេះដូងត្រូវបានណែនាំយ៉ាងខ្លាំង។",
        "cp_asymp": "🫀 **លំនាំនៃការឈឺទ្រូងដែលគ្មានរោគសញ្ញា** — ជំងឺកង្វះឈាមដោយស្ងៀមស្ងាត់គឺពិបាករកឃើញណាស់។ ការពិនិត្យបេះដូងសកម្ម (ECG, echo) ត្រូវបានណែនាំ។",
        "cp_typ": "⚠️ **បានរកឃើញការឈឺទ្រូងធម្មតា** — កំណត់ពេលពិនិត្យជំងឺបេះដូងដើម្បីវាយតម្លៃហានិភ័យនៃជំងឺសរសៃឈាមបេះដូង។",
        "vessels_multi": "🔬 **បានកត់សម្គាល់សរសៃឈាមដែលស្ទះច្រើន** ({} សរសៃឈាម) — នេះគឺជាសូចនាករដ៏រឹងមាំនៃជំងឺសរសៃឈាមបេះដូង។ ការពិនិត្យដោយអ្នកឯកទេសជាបន្ទាន់គឺចាំបាច់។",
        "vessels_one": "🔬 **សរសៃឈាមដែលស្ទះមួយ** — តាមដានយ៉ាងដិតដល់ និងពិភាក្សាអំពីជម្រើសនៃការថតសរសៃឈាមជាមួយគ្រូពេទ្យបេះដូងរបស់អ្នក។",
        "thal_rev": "🩺 **ពិការភាពដែលអាចត្រឡប់មកវិញបាននៅក្នុងតេស្តលំហូរឈាម** — នេះបង្ហាញពីកង្វះឈាមដែលបណ្តាលមកពីភាពតានតឹង។ ការពិនិត្យឡើងវិញនូវតេស្តភាពតានតឹងគឺចាំបាច់។",
        "thal_fixed": "🩺 **ពិការភាពថេរនៅក្នុងតេស្តលំហូរឈាម** — អាចបង្ហាញពីការស្ទះសរសៃឈាមពីមុន។ សូមពិភាក្សាជាមួយគ្រូពេទ្យរបស់អ្នក។",
        "age_risk": "📅 **ការពិនិត្យបេះដូងប្រចាំឆ្នាំគឺចាំបាច់** — ហានិភ័យកើនឡើងទៅតាមអាយុ។ ត្រូវប្រាកដថាមានការត្រួតពិនិត្យ ECG កម្រិតខ្លាញ់ និងសម្ពាធឈាមជាប្រចាំ។",
        "smoke": "🚭 **ជៀសវាងការជក់បារី** — ថ្នាំជក់គឺជាកត្តាហានិភ័យនាំមុខគេសម្រាប់សរសៃឈាមបេះដូងដែលអាចផ្លាស់ប្តូរបាន។",
        "sleep": "😴 **ផ្តល់អាទិភាពដល់ការគេង** — មានបំណងគេង 7–8 ម៉ោងក្នុងមួយយប់។ ការគេងមិនបានគ្រប់គ្រាន់បង្កើនហានិភ័យជំងឺបេះដូង។"
    }
}

# ─── Initialize App ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts heart disease risk from 14 clinical features using 5 ML models.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Paths and Model Loader ───────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_DIR  = BASE_DIR / "backend" / "model"

MODELS = {}
SCALER = None

MODEL_FILES = {
    "logistic":  "logistic_regression_model.pkl",
    "rf":        "random_forest_model.pkl",
    "svm":       "svm_model.pkl",
    "xgboost":   "xgboost_model.pkl",
    "lightgbm":  "lightgbm_model.pkl",
}

NEEDS_SCALING = {"logistic", "svm"}

def load_models():
    global SCALER
    for key, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if path.exists():
            MODELS[key] = joblib.load(path)
            print(f"OK: Loaded {key}: {filename}")
        else:
            print(f"WARN: Model not found: {path}")
    
    scaler_path = MODEL_DIR / "scaler.pkl"
    if scaler_path.exists():
        SCALER = joblib.load(scaler_path)
        print("OK: Loaded feature standardizer: scaler.pkl")

load_models()

# Exactly 14 features — matches training notebook FEATURES list in order
FEATURE_COLUMNS = [
    "Age", "Gender", "Chest_Pain_Type", "Resting_Blood_Pressure", "Cholesterol",
    "Fasting_Blood_Sugar", "Resting_ECG_Results", "Maximum_Heart_Rate",
    "Exercise_Induced_Angina", "Depression_Induced_By_Exercise",
    "Slope_Of_Peak_Exercise", "Major_Vessels_Colored_By_Fluoroscopy",
    "Thalassemia", "Risk_Score"
]

# ─── Input and Output Schemas ─────────────────────────────────────────────────
class TTSRequest(BaseModel):
    text: str
    lang: str

class PredictionInput(BaseModel):
    age:               float = Field(..., ge=20, le=100)
    gender:            int   = Field(..., ge=0, le=1)
    chest_pain_type:   int   = Field(..., ge=1, le=4)
    resting_bp:        float = Field(..., ge=80, le=220)
    cholesterol:       float = Field(..., ge=100, le=600)
    fasting_bs:        int   = Field(..., ge=0, le=1)
    resting_ecg:       int   = Field(..., ge=0, le=2)
    max_heart_rate:    float = Field(..., ge=60, le=220)
    exercise_angina:   int   = Field(..., ge=0, le=1)
    oldpeak:           float = Field(..., ge=0, le=10)
    st_slope:          int   = Field(..., ge=1, le=3)
    ca:                int   = Field(..., ge=0, le=3)
    thal:              int   = Field(..., ge=3, le=7)
    model:             str   = Field("logistic")
    lang:              str   = Field("en")

    @root_validator(pre=True)
    def check_model_keys(cls, values):
        m = values.get("model", "logistic")
        if m not in MODEL_FILES:
            raise ValueError(f"Model must be one of {list(MODEL_FILES.keys())}")
        return values

class PredictionOutput(BaseModel):
    model_used:      str
    risk_score:      float
    risk_percent:    int
    prediction:      int
    risk_level:      str
    verdict:         str
    recommendations: list[str]

# ─── Recommendations Builder ──────────────────────────────────────────────────
def build_recommendations(inp: PredictionInput, prediction: int, risk_score: float) -> list[str]:
    recs = []
    lang = inp.lang if inp.lang in TRANSLATIONS else "en"
    t = TRANSLATIONS[lang]

    if prediction == 1 or risk_score >= 0.5:
        recs.append(t["rec_high_risk"])
    else:
        recs.append(t["rec_low_risk"])

    if inp.resting_bp >= 140:
        recs.append(t["bp_high"].format(inp.resting_bp))
    elif inp.resting_bp >= 120:
        recs.append(t["bp_elevated"].format(inp.resting_bp))

    if inp.cholesterol >= 240:
        recs.append(t["chol_high"].format(inp.cholesterol))
    elif inp.cholesterol >= 200:
        recs.append(t["chol_borderline"].format(inp.cholesterol))

    if inp.fasting_bs == 1:
        recs.append(t["bs_high"])

    if inp.max_heart_rate < 120:
        recs.append(t["hr_low"].format(inp.max_heart_rate))

    if inp.exercise_angina == 1:
        recs.append(t["angina"])

    if inp.oldpeak >= 2.0:
        recs.append(t["st_dep"].format(inp.oldpeak))

    if inp.chest_pain_type == 4:
        recs.append(t["cp_asymp"])
    elif inp.chest_pain_type == 1:
        recs.append(t["cp_typ"])

    if inp.ca >= 2:
        recs.append(t["vessels_multi"].format(inp.ca))
    elif inp.ca == 1:
        recs.append(t["vessels_one"])

    if inp.thal == 7:
        recs.append(t["thal_rev"])
    elif inp.thal == 6:
        recs.append(t["thal_fixed"])

    if inp.age >= 60:
        recs.append(t["age_risk"])

    if prediction == 0 and risk_score < 0.35:
        recs.append(t["smoke"])
        recs.append(t["sleep"])

    return recs

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/tts", tags=["Voice"])
def text_to_speech(req: TTSRequest):
    try:
        tts = gTTS(text=req.text, lang=req.lang)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return Response(content=fp.getvalue(), media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "models_loaded": list(MODELS.keys())}

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(inp: PredictionInput):
    model_key = inp.model
    if model_key not in MODELS:
        raise HTTPException(status_code=503, detail=f"Model '{model_key}' is currently unavailable.")

    model = MODELS[model_key]

    # Risk_Score: engineered feature used during training
    # Formula: (age × cholesterol / 1000) + (resting_bp / 100)
    risk_score_feature = (inp.age * inp.cholesterol / 1000) + (inp.resting_bp / 100)

    # Build feature vector in exact training order (14 features)
    X = np.array([[
        inp.age,            # Age
        inp.gender,         # Gender
        inp.chest_pain_type,# Chest_Pain_Type
        inp.resting_bp,     # Resting_Blood_Pressure
        inp.cholesterol,    # Cholesterol
        inp.fasting_bs,     # Fasting_Blood_Sugar
        inp.resting_ecg,    # Resting_ECG_Results
        inp.max_heart_rate, # Maximum_Heart_Rate
        inp.exercise_angina,# Exercise_Induced_Angina
        inp.oldpeak,        # Depression_Induced_By_Exercise
        inp.st_slope,       # Slope_Of_Peak_Exercise
        inp.ca,             # Major_Vessels_Colored_By_Fluoroscopy
        inp.thal,           # Thalassemia
        risk_score_feature, # Risk_Score
    ]], dtype=float)

    # Logistic Regression and SVM were trained on scaled data
    X_transformed = X.copy()
    if model_key in NEEDS_SCALING and SCALER is not None:
        try:
            X_transformed = SCALER.transform(X)
        except Exception:
            pass  # fallback to unscaled if scaler dimensions differ

    try:
        prediction = int(model.predict(X_transformed)[0])
        if hasattr(model, "predict_proba"):
            risk_score = float(model.predict_proba(X_transformed)[0][1])
        else:
            df = model.decision_function(X_transformed)[0]
            risk_score = float(1 / (1 + np.exp(-df)))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction execution error: {exc}")

    lang = inp.lang if inp.lang in TRANSLATIONS else "en"
    t = TRANSLATIONS[lang]

    if risk_score < 0.35:
        risk_level = t["risk_level_low"]
        verdict    = t["verdict_lower"]
    elif risk_score < 0.60:
        risk_level = t["risk_level_moderate"]
        verdict    = t["verdict_moderate"]
    else:
        risk_level = t["risk_level_high"]
        verdict    = t["verdict_high"]

    recommendations = build_recommendations(inp, prediction, risk_score)

    return PredictionOutput(
        model_used      = model_key,
        risk_score      = round(risk_score, 4),
        risk_percent    = round(risk_score * 100),
        prediction      = prediction,
        risk_level      = risk_level,
        verdict         = verdict,
        recommendations = recommendations,
    )
