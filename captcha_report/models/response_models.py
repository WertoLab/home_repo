import typing as tp

from datetime import date
from pydantic import BaseModel


class CaptchaReportResponse(BaseModel):
    class Report(BaseModel):
        report_date: date
        information: str | None

    report_count: int
    reports: tp.List[Report]
