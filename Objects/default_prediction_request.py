from typing import Union
from pydantic import BaseModel


class LoanDefaultPredictionRequest(BaseModel):
    loan: int
    mortdue: Union[float, None]
    value: Union[float, None]
    reason: Union[str, None]
    job: Union[str, None]
    yoj: Union[float, None]
    derog: Union[float, None]
    delinq: Union[float, None]
    clage: Union[float, None]
    ninq: Union[float, None]
    clno: Union[float, None]
    debtinc: Union[float, None]
