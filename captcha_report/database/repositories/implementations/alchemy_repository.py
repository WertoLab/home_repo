import typing as tp
from captcha_report.config import settings
from captcha_report.database.connection import SqlAlchemyConnection
from captcha_report.database.repositories.abstractions import CaptchaReportRepository
from kink import inject


@inject(alias=CaptchaReportRepository)
class SqlAlchemyCaptchaReportRepository(CaptchaReportRepository):
    _connection: SqlAlchemyConnection

    def __init__(self) -> None:
        self._connection = SqlAlchemyConnection(
            database_url=settings.database_url,
            echo=settings.echo,
        )
