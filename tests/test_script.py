from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status":"ok"}

def test_predict_empty_input():
    payload = {
        "input_array" : [[]]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_valid_data():
    payload = {
        "input_array" : [[2002, 3, 5, 7, 8, 1, 3]]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body

