from typing import Union
from pydantic import BaseModel


class LoanDefaultPredictionRequest(BaseModel):
    loan: int
    mortdue: Union[float, None] = None
    value: Union[float, None] = None
    reason: Union[str, None] = None
    job: Union[str, None] = None
    yoj: Union[float, None] = None
    derog: Union[float, None] = None
    delinq: Union[float, None] = None
    clage: Union[float, None] = None
    ninq: Union[float, None] = None
    clno: Union[float, None] = None
    debtinc: Union[float, None] = None
