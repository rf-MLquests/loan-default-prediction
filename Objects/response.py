from pydantic import BaseModel


class Response(BaseModel):
    likelyToDefault: int
    probabilityToDefault: float
