from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import joblib


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "credit_card_fraud_model.pkl"
FRONTEND_DIR = BASE_DIR / "frontend"


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = joblib.load(MODEL_PATH)


# =========================================================
# CREATE FASTAPI APP
# =========================================================

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for detecting fraudulent credit card transactions",
    version="1.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# INPUT DATA MODEL
# =========================================================

class Transaction(BaseModel):
    Time: float

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Random Forest",
        "message": "Credit Card Fraud Detection API is running"
    }


# =========================================================
# PREDICTION API
# =========================================================

@app.post("/predict")
def predict(transaction: Transaction):

    # Convert input to dictionary
    data = transaction.model_dump()

    # Convert dictionary to DataFrame
    input_data = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get probability
    probability = model.predict_proba(input_data)[0][1]

    # Convert prediction to result
    if prediction == 1:
        result = "Fraudulent Transaction"
    else:
        result = "Legitimate Transaction"

    return {
        "prediction": int(prediction),
        "result": result,
        "fraud_probability": round(float(probability), 4)
    }


# =========================================================
# SERVE FRONTEND
# =========================================================

if FRONTEND_DIR.exists():

    # Serve frontend files directly from the root URL
    # This must be AFTER all API routes.

    app.mount(
        "/",
        StaticFiles(
            directory=FRONTEND_DIR,
            html=True
        ),
        name="frontend"
    )
