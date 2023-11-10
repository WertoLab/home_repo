import typing as tp
from captcha_report.database.connection.alchemy_connection import SqlAlchemyConnection
from captcha_report.database.repositories.abstractions import CaptchaReportRepository
from kink import inject


@inject(alias=CaptchaReportRepository)
class SqlAlchemyCaptchaReportRepository(CaptchaReportRepository):
    _connection: SqlAlchemyConnection

    def __init__(self, connection: SqlAlchemyConnection) -> None:
        self._connection = connection
