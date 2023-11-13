import typing as tp
from datetime import date
from pydantic import BaseModel
from captcha_report.models.captcha_report_models import StatusEnum


class ReportPaginationParams(BaseModel):
    limit: int | None = None
    offset: int = 0


class ReportGetParams(ReportPaginationParams):
    iso_date: date
    status: StatusEnum

    def get_pagination(self) -> ReportPaginationParams:
        return ReportPaginationParams(
            limit=self.limit,
            offset=self.offset,
        )
