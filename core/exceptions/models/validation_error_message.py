from __future__ import annotations
import typing as tp

from pydantic import BaseModel


class ValidationErrorMessage(BaseModel):
    fields: tp.Tuple[str]
    message: str
    input_value: str
