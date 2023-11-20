import typing as tp
from datetime import date
from uuid import UUID
from core.config import settings
from captcha_report.database.connection import SqlAlchemyConnection
from captcha_report.database.repositories import CaptchaReportRepository
from kink import inject
from sqlalchemy import select, func
from sqlalchemy.sql import text

from captcha_report.database.models import CaptchaReport
from captcha_report.models.param_models import (
    ReportPaginationParams,
    StatisticTimeInterval,
)
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
            select(func.count())
            .select_from(CaptchaReport)
            .where(
                CaptchaReport.report_date == report_date,
                CaptchaReport.status == status,
            )
        )

        async with self._connection.session_factory() as session:
            count = await session.scalar(query)
            return count

    async def count_reports_by_datetime_and_status(
        self,
        report_date: date,
        status: StatusEnum,
        time_interval: StatisticTimeInterval,
    ) -> int:
        query = (
            select(func.count())
            .select_from(CaptchaReport)
            .where(
                CaptchaReport.status == status,
                CaptchaReport.report_date == report_date,
                CaptchaReport.report_time >= time_interval.start_time,
                CaptchaReport.report_time <= time_interval.end_time,
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

    async def get_reports_by_date_and_status_with_pagination(
        self,
        report_date: date,
        status: StatusEnum,
        pagination: ReportPaginationParams,
    ):
        query = (
            select(CaptchaReport)
            .where(
                CaptchaReport.report_date == report_date,
                CaptchaReport.status == status,
            )
            .limit(pagination.limit)
            .offset(pagination.offset)
        )

        async with self._connection.session_factory() as session:
            reports = await session.scalars(query)
            return [report.to_domain_model() for report in reports]

    async def get_report_by_uuid(self, uuid: UUID) -> tp.Optional[CaptchaReportInDB]:
        query = select(CaptchaReport).where(CaptchaReport.uuid == uuid)
        async with self._connection.session_factory() as session:
            report = await session.scalar(query)
            if report is not None:
                return report.to_domain_model()

    async def get_all_errors(
        self,
        report_date: date,
        pagination: ReportPaginationParams,
        time_interval: StatisticTimeInterval,
    ):
        query = """SELECT DISTINCT information ->> 'errors' FROM captcha_report
                        WHERE report_date = :report_date
                            AND 
                        report_time BETWEEN :start_time AND :end_time
                    LIMIT :limit OFFSET :offset;"""

        async with self._connection.session_factory() as session:
            errors = await session.execute(
                text(query),
                {
                    "report_date": report_date,
                    "start_time": time_interval.start_time,
                    "end_time": time_interval.end_time,
                    "limit": pagination.limit,
                    "offset": pagination.offset,
                },
            )

            return [error[0] for error in errors.fetchall()]

    async def save_report(self, report: CaptchaReportInDB):
        report = CaptchaReport(**report.model_dump())
        async with self._connection.session_factory() as session:
            session.add(report)
            await session.commit()
