from pydantic import BaseModel


class PredictionRequest(BaseModel):
    LIMIT_BAL: float
    AGE: int
    PAY_0: float
    PAY_2: float
    PAY_3: float
    PAY_4: float    
    PAY_5: float
    PAY_6: float
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float
    EDUCATION: int
    MARRIAGE: int
    SEX: int

class PredictionResponse(BaseModel):
    prediction: int
    probability: float