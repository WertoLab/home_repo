import typing as tp
from captcha_report.database.repositories.abstractions import CaptchaReportRepository
from kink import inject


@inject
class CaptchaReportService:
    _repository: CaptchaReportRepository

    def __init__(self, repository: CaptchaReportRepository) -> None:
        self._repository = repository
