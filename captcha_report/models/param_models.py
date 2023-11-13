import typing as tp
from datetime import date, datetime, time
from pydantic import BaseModel
from captcha_report.models.captcha_report_models import StatusEnum


class ReportPaginationParams(BaseModel):
    limit: int
    offset: int


class ReportGetParams(ReportPaginationParams):
    iso_date: date
    status: StatusEnum

    limit: int | None = None
    offset: int = 0

    def get_pagination(self) -> ReportPaginationParams:
        return ReportPaginationParams(
            limit=self.limit,
            offset=self.offset,
        )


class StatisticTimeInterval(BaseModel):
    start_time: time
    end_time: time


class StatisticDatetimeParams(BaseModel):
    iso_date: date
    utc_start_time: time | None = None
    utc_end_time: time = datetime.utcnow().strftime("%H:%M:%S")

    def get_time_interval(self) -> StatisticTimeInterval:
        return StatisticTimeInterval(
            start_time=self.utc_start_time,
            end_time=self.utc_end_time,
        )
