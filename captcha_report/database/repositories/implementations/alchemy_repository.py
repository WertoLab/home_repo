import typing as tp
from datetime import date
from captcha_report.config import settings
from captcha_report.database.connection import SqlAlchemyConnection
from captcha_report.database.repositories import CaptchaReportRepository
from kink import inject
from sqlalchemy import select, func

from captcha_report.database.models import CaptchaReport
from captcha_report.models.captcha_report_models import StatusEnum, CaptchaReportInDB


@inject(alias=CaptchaReportRepository)
class SqlAlchemyCaptchaReportRepository(CaptchaReportRepository):
    _connection: SqlAlchemyConnection

    def __init__(self) -> None:
        self._connection = SqlAlchemyConnection(
            database_url=settings.database_url,
            echo=settings.echo,
        )

    async def count_reports_by_date_and_status(
        self, report_date: date, status: StatusEnum
    ):
        query = (
            select(func.count("*"))
            .select_from(CaptchaReport)
            .where(
                CaptchaReport.report_date == report_date,
                CaptchaReport.status == status,
            )
        )

        async with self._connection.session_factory() as session:
            count = await session.scalar(query)
            return count

    async def get_reports_by_date_and_status(
        self, report_date: date, status: StatusEnum
    ):
        query = select(CaptchaReport).where(
            CaptchaReport.report_date == report_date,
            CaptchaReport.status == status,
        )

        async with self._connection.session_factory() as session:
            reports = await session.scalars(query)
            return [report.to_domain_model() for report in reports]

    async def save_report(self, report: CaptchaReportInDB):
        report = CaptchaReport(**report.model_dump())
        async with self._connection.session_factory() as session:
            session.add(report)
            await session.commit()
