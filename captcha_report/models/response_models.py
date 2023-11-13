import typing as tp

from datetime import date
from pydantic import BaseModel
from captcha_report.models.captcha_report_models import CaptchaReportInformation


class CaptchaReportResponse(BaseModel):
    class Report(BaseModel):
        report_date: date
        information: CaptchaReportInformation | None

    report_count: int
    reports: tp.List[Report]
