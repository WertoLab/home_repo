import typing as tp
from abc import abstractmethod
from datetime import date
from captcha_report.models.captcha_report_models import CaptchaReportInDB, StatusEnum


class CaptchaReportRepository(tp.Protocol):
    @abstractmethod
    async def count_reports_by_date_and_status(
        self, report_date: date, status: StatusEnum
    ) -> int:
        ...

    @abstractmethod
    async def get_reports_by_date_and_status(
        self, report_date: date, status: StatusEnum
    ) -> tp.List[CaptchaReportInDB]:
        ...

    @abstractmethod
    async def save_report(self, report: CaptchaReportInDB) -> None:
        ...
