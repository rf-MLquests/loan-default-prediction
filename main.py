import Production.dt_service as dt
import Production.rf_service as rf
import Production.xgb_service as xgb
from Objects.default_prediction_request import LoanDefaultPredictionRequest
from fastapi import FastAPI
from Inference.predict import deserialize_response

app = FastAPI()


@app.post("/myLoan/default-risk-dt")
async def default_risk_with_dt(request: LoanDefaultPredictionRequest):
    upper_case_dict = {k.upper(): v for k, v in request.dict().items()}
    labels, probabilities = dt.predict_with_dt(upper_case_dict)
    return deserialize_response(labels, probabilities)


@app.post("/myLoan/default-risk-rf")
async def default_risk_with_rf(request: LoanDefaultPredictionRequest):
    upper_case_dict = {k.upper(): v for k, v in request.dict().items()}
    labels, probabilities = rf.predict_with_rf(upper_case_dict)
    return deserialize_response(labels, probabilities)


@app.post("/myLoan/default-risk-xgb")
async def default_risk_with_xgb(request: LoanDefaultPredictionRequest):
    upper_case_dict = {k.upper(): v for k, v in request.dict().items()}
    labels, probabilities = xgb.predict_with_xgb(upper_case_dict)
    return deserialize_response(labels, probabilities)
