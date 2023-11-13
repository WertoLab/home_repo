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


class CaptchaReportInformation(BaseModel):
    request_body: tp.Dict[str, tp.Any]
    errors: tp.List[tp.Dict[str, tp.Any]] | str | None = Field(default=None)


class CaptchaReportInDB(PrimaryKey):
    report_date: date
    status: StatusEnum
    information: tp.Dict[str, tp.Any] | None = Field(default=None)

    class Config:
        from_attributes = True


class CaptchaReportStatistic(BaseModel):
    total: int
    failed: int
    resolved: int
    not_resolved: int


class CaptchaReportCreate(BaseModel):
    status: StatusEnum
    information: CaptchaReportInformation | None = Field(default=None)


class CaptchaReportFilter(BaseModel):
    status: StatusEnum
    report_date: date
