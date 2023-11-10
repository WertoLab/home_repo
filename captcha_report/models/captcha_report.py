import typing as tp
from pydantic import BaseModel, UUID4, Field
from datetime import datetime
from enum import Enum


class StatusEnum(str, Enum):
    RESOLVED = "resolved"
    FAILED = "failed"
    NOT_RESOLVED = "not resolved"


class PrimaryKey(BaseModel):
    uuid: UUID4


class CaptchaReport(BaseModel):
    resolve_datetime: datetime
    status: StatusEnum
    information: str = Field(max_length=500)
