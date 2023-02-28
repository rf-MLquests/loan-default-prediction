from pydantic import BaseModel


class Response(BaseModel):
    bad: int
    goodProb: float
    badProb: float
