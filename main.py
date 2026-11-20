from pathlib import Path
from typing import Callable
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
CHAMPION_PATH = ROOT / "models" / "housing_champion.joblib"

class HousingInput(BaseModel):
    area: float = Field(gt=0)
    bedrooms: int = Field(ge=1)
    bathrooms: int = Field(ge=1)
    stories: int = Field(ge=1)
    mainroad: str
    guestroom: str
    basement: str
    hotwaterheating: str
    airconditioning: str

def load_champion():
    return joblib.load(CHAMPION_PATH)

def create_app(model_loader: Callable = load_champion):
    app = FastAPI(
        title="Housing Price Prediction API",
        description="Housing price prediction API with a locally saved champion model.",
        version="4.0.0"
    )

    @app.on_event("startup")
    async def load_model():
        app.state.model = model_loader()

    @app.get("/")
    def ready():
        return {"status": "Ready to serve"}

    @app.get("/health")
    def health(request: Request):
        return {"status": "healthy", "model_name": "housing-price-regressor",
                "model_path": str(CHAMPION_PATH)}

    @app.post("/predict")
    def predict(data: HousingInput, request: Request):
        payload = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        prediction = float(request.app.state.model.predict(pd.DataFrame([payload]))[0])
        return {"predicted_price": round(prediction, 2)}

    return app

app = create_app()
