from pydantic import BaseModel


class PredictionRequest(BaseModel):
    LIMIT_BAL: float
    AGE: int
    PAY_1: float
    BILL_AMT1: float
    PAY_AMT1: float
    EDUCATION: int
    MARRIAGE: int
    SEX: int

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_category: str