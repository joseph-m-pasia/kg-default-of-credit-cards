from fastapi import FastAPI
import joblib

app = FastAPI()

model = None


def get_model():
    global model

    if model is None:
        model = joblib.load("artifacts/model.pkl")

    return model


@app.post("/predict")
def predict(data: dict):

    model = get_model()

    prediction = model.predict([data])

    return {
        "prediction": int(prediction[0])
    }