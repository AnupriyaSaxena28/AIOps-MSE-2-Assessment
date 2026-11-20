from fastapi.testclient import TestClient
from main import create_app

class FakeHousingModel:
    def predict(self, rows):
        return [5250000.0]

def test_health():
    app = create_app(model_loader=FakeHousingModel)
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

def test_housing_prediction():
    app = create_app(model_loader=FakeHousingModel)
    payload = {
        "area": 7420, "bedrooms": 4, "bathrooms": 2, "stories": 3,
        "mainroad": "yes", "guestroom": "no", "basement": "no",
        "hotwaterheating": "no", "airconditioning": "yes"
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json()["predicted_price"] == 5250000.0

def test_invalid_housing_input():
    app = create_app(model_loader=FakeHousingModel)
    payload = {
        "area": -1, "bedrooms": 4, "bathrooms": 2, "stories": 3,
        "mainroad": "yes", "guestroom": "no", "basement": "no",
        "hotwaterheating": "no", "airconditioning": "yes"
    }
    with TestClient(app) as client:
        assert client.post("/predict", json=payload).status_code == 422
