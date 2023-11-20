import typing as tp
from abc import abstractmethod
from datetime import date
from uuid import UUID

from captcha_report.models.param_models import (
    ReportPaginationParams,
    StatisticTimeInterval,
)
from captcha_report.models.captcha_report_models import CaptchaReportInDB, StatusEnum


class CaptchaReportRepository(tp.Protocol):
    @abstractmethod
    async def count_reports_by_date_and_status(
        self, report_date: date, status: StatusEnum
    ) -> int:
        ...

    @abstractmethod
    async def count_reports_by_datetime_and_status(
        self,
        report_date: date,
        status: StatusEnum,
        time_interval: StatisticTimeInterval,
    ) -> int:
        ...

    @abstractmethod
    async def get_reports_by_date_and_status(
        self, report_date: date, status: StatusEnum
    ) -> tp.List[CaptchaReportInDB]:
        ...

    @abstractmethod
    async def get_reports_by_date_and_status_with_pagination(
        self,
        report_date: date,
        status: StatusEnum,
        pagination: ReportPaginationParams,
    ) -> tp.List[CaptchaReportInDB]:
        ...

    @abstractmethod
    async def get_report_by_uuid(self, uuid: UUID) -> tp.Optional[CaptchaReportInDB]:
        ...

    @abstractmethod
    async def get_all_errors(
        self,
        report_date: date,
        time_interval: StatisticTimeInterval,
        pagination: ReportPaginationParams,
    ) -> tp.List[str]:
        ...

    @abstractmethod
    async def save_report(self, report: CaptchaReportInDB) -> None:
        ...
