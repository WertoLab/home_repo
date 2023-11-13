from datetime import date, time
import typing as tp

from pydantic import UUID4, BaseModel, model_validator, Field
from captcha_report.models.captcha_report_models import CaptchaReportInDB, StatusEnum


class CaptchaReportListResponse(BaseModel):
    class ReportDetail(BaseModel):
        uuid: UUID4
        report_date: date
        report_time: time
        status: StatusEnum
        errors: tp.List[tp.Dict[str, tp.Any]] | str | None = Field(default=None)

        @model_validator(mode="before")
        def exclude_request_body(data: CaptchaReportInDB):
            print(data)
            report = data.model_dump()
            if isinstance(data.information, dict):
                report["errors"] = data.information.get("errors")

            return report

    total_count: int
    reports: tp.List[ReportDetail]
