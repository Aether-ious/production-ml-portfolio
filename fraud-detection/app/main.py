from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
from .schema import Transaction
from .model import FraudModel
from .monitoring import generate_drift_report
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(
    title="Production Fraud Detection API",
    description="Real-time fraud detection with drift monitoring",
    version="1.0.0"
)

# Prometheus metrics
REQUEST_COUNT = Counter("fraud_requests_total", "Total fraud API requests", ["endpoint"])
PREDICTION_LATENCY = Histogram("fraud_prediction_latency_seconds", "Prediction latency")
FRAUD_PREDICTIONS = Counter("fraud_predictions_total", "Fraudulent predictions", ["predicted"])

model = FraudModel()

@app.on_event("startup")
async def startup_event():
    print("Model loaded successfully")

@app.post("/predict")
async def predict(transaction: Transaction):
    REQUEST_COUNT.labels(endpoint="/predict").inc()
    
    with PREDICTION_LATENCY.time():
        try:
            df = pd.DataFrame([transaction.dict()])
            prob = float(model.predict_proba(df)[0])
            is_fraud = prob > 0.5
            
            if is_fraud:
                FRAUD_PREDICTIONS.labels(predicted="fraud").inc()
            else:
                FRAUD_PREDICTIONS.labels(predicted="legit").inc()
                
            return {"fraud_probability": round(prob, 4), "is_fraud": is_fraud}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

@app.get("/drift-report")
async def drift_report():
    # Use last 1000 requests or a sample — here we fake it with reference data
    # In real prod you would collect inference data
    sample = pd.read_csv("data/train_sample.csv").head(500)
    path = generate_drift_report(sample)
    return FileResponse(path)

@app.get("/metrics")
async def metrics():
    return JSONResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)