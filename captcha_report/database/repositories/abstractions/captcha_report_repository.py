import typing as tp
from abc import abstractmethod


class CaptchaReportRepository(tp.Protocol):
    @abstractmethod
    async def get_reports_by_date(self, date) -> ...:
        ...

    @abstractmethod
    async def save_report(self, captcha_report: ...) -> None:
        ...
