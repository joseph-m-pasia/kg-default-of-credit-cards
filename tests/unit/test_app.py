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

    payload = {
        "LIMIT_BAL": 20000,
        "AGE": 35    
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200
