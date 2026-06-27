from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

# Fixture
client = TestClient(app)

@patch("app.main.get_model")    
def test_predict(mock_get_model):
    '''
    When you send a valid JSON payload to /predict, the API responds with HTTP 200 OK.
    This confirms that:
    - the endpoint exists
    - FastAPI can parse the input
    - the model loads correctly
    - the endpoint does not crash
    It’s a smoke test — a minimal test to ensure the endpoint is alive.
    '''

    fake_model = mock_get_model.return_value
    fake_model.predict.return_value = [1]

    payload = PredictionRequest(
        LIMIT_BAL=20000,
        AGE=35,
        PAY_0=1.0,
        PAY_2=2.0,
        PAY_3=3.0,
        PAY_4=4.0,
        PAY_5=5.0,
        PAY_6=6.0,
        BILL_AMT1=1000.0,
        BILL_AMT2=2000.0,
        BILL_AMT3=3000.0,
        BILL_AMT4=4000.0,
        BILL_AMT5=5000.0,
        BILL_AMT6=6000.0,
        PAY_AMT1=100.0,
        PAY_AMT2=200.0,
        PAY_AMT3=300.0,
        PAY_AMT4=400.0,
        PAY_AMT5=500.0,
        PAY_AMT6=600.0,
        EDUCATION=2,
        MARRIAGE=1,
        SEX=1
    )

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == 1