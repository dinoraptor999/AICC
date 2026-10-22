"""Mock API for frontend testing without loading trained models."""

import random
import time
from typing import Any

from fastapi import FastAPI, File, UploadFile


app = FastAPI(
    title="Traffic Congestion Prediction API (Mock)",
    version="1.0.0",
)


@app.post("/api/v1/predict")
def predict_video(file: UploadFile = File(...)) -> dict[str, Any]:
    """Return sample congestion predictions after a three-second delay."""
    del file
    time.sleep(3)

    return {
        "status": "success",
        "model_version": "v1.0-car-only",
        "results": [
            {
                "timestamp": 10.5,
                "congestion_level": random.randint(0, 3),
                "features": {
                    "car_count": random.randint(0, 30),
                    "avg_speed": round(random.uniform(0, 40), 1),
                },
            }
        ],
    }
