import Production.dt_service as dt
import Production.rf_service as rf
import Production.xgb_service as xgb
from Objects import LoanDefaultPredictionRequest
from fastapi import FastAPI

app = FastAPI()


@app.post("/myLoan/default-risk-dt")
async def default_risk_with_dt(request: LoanDefaultPredictionRequest):
    return dt.predict_with_dt(request)


@app.post("/myLoan/default-risk-rf")
async def default_risk_with_rf(request: LoanDefaultPredictionRequest):
    return rf.predict_with_rf(request)


@app.post("/myLoan/default-risk-xgb")
async def default_risk_with_xgb(request: LoanDefaultPredictionRequest):
    return xgb.predict_with_xgb(request)
