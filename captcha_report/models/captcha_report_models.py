import typing as tp
from pydantic import BaseModel, UUID4, Field
from datetime import date
from enum import Enum


class StatusEnum(str, Enum):
    RESOLVED = "resolved"
    FAILED = "failed"
    NOT_RESOLVED = "not_resolved"


class PrimaryKey(BaseModel):
    uuid: UUID4


class CaptchaReportInDB(PrimaryKey):
    report_date: date
    status: StatusEnum
    information: str | None = Field(max_length=500, default=None)

    class Config:
        from_attributes = True


class CaptchaReportStatistic(BaseModel):
    total: int
    failed: int
    resolved: int
    not_resolved: int


class CaptchaReportCreate(BaseModel):
    status: StatusEnum
    information: str | None = Field(max_length=500, default=None)


class CaptchaReportFilter(BaseModel):
    status: StatusEnum
    report_date: date
