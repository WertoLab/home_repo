import typing as tp
from pydantic import BaseModel, model_validator
from captcha_report.models.captcha_report_models import CaptchaReportInDB


class CaptchaReportResponse(BaseModel):
    report_count: int
    reports: tp.List[CaptchaReportInDB]
