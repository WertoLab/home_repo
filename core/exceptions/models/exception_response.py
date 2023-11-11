import typing as tp
from pydantic import BaseModel


class ExceptionResponse(BaseModel):
    status_code: int
    message: str
