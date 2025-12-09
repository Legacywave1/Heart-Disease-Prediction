import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, List
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline
import logging

app = FastAPI(
    title='Heart Disease Risk Predictor API',
    description ='Clinical-grade heart disease risk assessment with explainability',
    version='2.0',
    docs_url='/docs',
    redoc_url='/redoc'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production change to ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    predictor = PredictPipeline()
    logging.info('Prediction pipeline loaded successfully')
except Exception as e:
    logging.error(f'Failed to load model: {e}')
    raise

class PatientInput(BaseModel):
    Age: int = Field(..., ge= 1, le = 120, description='Age in years')
    Sex: Literal['Male', 'Female']
    ChestPainType: Literal['TA', 'ATA', 'NAP', 'ASY']  = Field(
        ..., description = 'TA = Typical Angina, ATA = Atypical Angina, NAP = Non-Angina Pain, ASY = Asymptomatic'
    )
    RestingBP: int = Field(..., ge = 80, le= 200, description = 'Resting Blood Pressure (mmHg')
    Cholesterol: int = Field(..., ge = 100, le = 600, description = 'Serum cholesterol (mg/dl)')
    FastingBS: int = Field(..., ge = 0, le = 1, description = '1 if fasting blood sugar > 120 mg/dl, 0 otherwise')
    RestingECG: Literal['Normal', 'ST', 'LVH']
    thalch: int = Field(..., ge = 60, le = 220, description = 'Maximum heart rate achieved')
    ExerciseAngina: Literal['Y', 'N']
    OldPeak: float = Field(..., ge = -2.0, le = 7.0, description ='ST depressiong induced by exercise')
    ST_Slope: Literal['Up', 'Flat', 'Down']

    class Config:
        schema_extra = {
            "example": {
                'Age': 55,
                'Sex': 'Male',
                'ChestPainType': 'NAP',
                'RestingBP': 140,
                'Cholesterol': 217,
                'FastingBS': 0,
                'RestingECG': 'Normal',
                'thalach': 111,
                'ExerciseAngina': 'Y',
                'OldPeak': 5.6,
                'ST_Slope': 'Flat'
            }
        }

class PredictionResponse(BaseModel):
    risk_level: str
    risk_probability: float
    top_contributors: List[dict]
    recommendation: str
    model_status: str

@app.get('/')
def home():
    return {
        'message': 'Heart Disease Risk Predictor API is Live',
        'health': '/health',
        'docs': '/docs',
        'predict': "Post /predict"
    }

@app.get('/health')
def health_check():
    return {'status': 'healthy',
            'model_loaded': True}


@app.post('/predict', response_model = PredictionResponse)
def predict_risk(patient: PatientInput):
    try:
        result = predictor.predict(patient.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f'Prediction Failed: {str(e)}')

if __name__ == '__main__':
    uvicorn.run("app.main:app", host='0.0.0.0', port = 8000, reload = True)
