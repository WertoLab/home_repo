import typing as tp
from datetime import date, time
from pydantic import BaseModel
from captcha_report.models.captcha_report_models import StatusEnum


class ReportPaginationParams(BaseModel):
    limit: int
    offset: int


class StatisticTimeInterval(BaseModel):
    start_time: time
    end_time: time


class PaginationMixin:
    def get_pagination(self) -> ReportPaginationParams:
        return ReportPaginationParams(
            limit=self.limit,
            offset=self.offset,
        )


class TimeIntervalMixin:
    def get_time_interval(self) -> StatisticTimeInterval:
        start_time, end_time = self.utc_time_interval.split("-")
        return StatisticTimeInterval(
            start_time=start_time.strip(),
            end_time=end_time.strip(),
        )


class ReportGetParams(PaginationMixin, ReportPaginationParams):
    iso_date: date
    status: StatusEnum

    limit: int | None = None
    offset: int = 0


class ErrorsGetParams(PaginationMixin, TimeIntervalMixin, ReportPaginationParams):
    iso_date: date
    utc_time_interval: str
    limit: int = 100
    offset: int = 0


class StatisticDatetimeParams(TimeIntervalMixin, BaseModel):
    iso_date: date
    utc_time_interval: str | None = None
