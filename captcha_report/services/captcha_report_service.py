import typing as tp
import asyncio

from uuid import UUID, uuid4
from captcha_report.database.repositories import CaptchaReportRepository
from kink import inject
from datetime import date, datetime
from captcha_report.models.param_models import (
    ReportPaginationParams,
    StatisticTimeInterval,
)
from captcha_report.models.captcha_report_models import (
    CaptchaReportStatistic,
    CaptchaReportCreate,
    CaptchaReportInDB,
    StatusEnum,
)


@inject
class CaptchaReportService:
    _repository: CaptchaReportRepository

    def __init__(self, repository: CaptchaReportRepository) -> None:
        self._repository = repository

    async def count_reports_statistic_by_date(
        self, report_date: date
    ) -> tp.List[CaptchaReportStatistic]:
        async with asyncio.TaskGroup() as tg:
            count_resolved = tg.create_task(
                self._repository.count_reports_by_date_and_status(
                    report_date=report_date,
                    status=StatusEnum.RESOLVED,
                )
            )

            count_not_resolved = tg.create_task(
                self._repository.count_reports_by_date_and_status(
                    report_date=report_date,
                    status=StatusEnum.NOT_RESOLVED,
                )
            )

            count_failed = tg.create_task(
                self._repository.count_reports_by_date_and_status(
                    report_date=report_date,
                    status=StatusEnum.FAILED,
                )
            )

        resolved, not_resolved, failed = (
            count_resolved.result(),
            count_not_resolved.result(),
            count_failed.result(),
        )

        return CaptchaReportStatistic(
            resolved=resolved,
            not_resolved=not_resolved,
            failed=failed,
            total=resolved + not_resolved + failed,
        )

    async def count_reports_statistic_by_datetime(
        self, report_date: date, time_interval: StatisticTimeInterval
    ) -> tp.List[CaptchaReportStatistic]:
        async with asyncio.TaskGroup() as tg:
            count_resolved = tg.create_task(
                self._repository.count_reports_by_datetime_and_status(
                    report_date=report_date,
                    status=StatusEnum.RESOLVED,
                    time_interval=time_interval,
                )
            )

            count_not_resolved = tg.create_task(
                self._repository.count_reports_by_datetime_and_status(
                    report_date=report_date,
                    status=StatusEnum.NOT_RESOLVED,
                    time_interval=time_interval,
                )
            )

            count_failed = tg.create_task(
                self._repository.count_reports_by_datetime_and_status(
                    report_date=report_date,
                    status=StatusEnum.FAILED,
                    time_interval=time_interval,
                )
            )

        resolved, not_resolved, failed = (
            count_resolved.result(),
            count_not_resolved.result(),
            count_failed.result(),
        )

        return CaptchaReportStatistic(
            resolved=resolved,
            not_resolved=not_resolved,
            failed=failed,
            total=resolved + not_resolved + failed,
        )

    async def get_reports_by_date_and_status(
        self, report_date: date, status: StatusEnum
    ) -> tp.List[CaptchaReportInDB]:
        reports = await self._repository.get_reports_by_date_and_status(
            report_date=report_date,
            status=status,
        )

        return reports

    async def get_failed_reports_by_datetime_and_status(
        self,
        report_date: date,
        time_interval: StatisticTimeInterval,
        pagination: ReportPaginationParams,
    ) -> tp.List[CaptchaReportInDB]:
        reports = (
            await self._repository.get_reports_by_datetime_and_status_with_pagination(
                report_date=report_date,
                time_interval=time_interval,
                pagination=pagination,
                status=StatusEnum.FAILED,
            )
        )

        return reports

    async def get_reports_by_date_and_status_with_count(
        self,
        report_date: date,
        status: StatusEnum,
        pagination: ReportPaginationParams,
    ) -> tp.Tuple[int, tp.List[CaptchaReportInDB]]:
        async with asyncio.TaskGroup() as tg:
            get_reports = tg.create_task(
                self._repository.get_reports_by_date_and_status_with_pagination(
                    report_date=report_date,
                    status=status,
                    pagination=pagination,
                )
            )

            count_reports = tg.create_task(
                self._repository.count_reports_by_date_and_status(
                    report_date=report_date,
                    status=status,
                )
            )

        return count_reports.result(), get_reports.result()

    async def get_report_by_uuid(self, uuid: UUID) -> tp.Optional[CaptchaReportInDB]:
        report = await self._repository.get_report_by_uuid(uuid)
        return report

    async def get_all_errors(
        self,
        report_date: date,
        time_interval: StatisticTimeInterval,
        pagination: ReportPaginationParams,
    ) -> tp.List[str]:
        errors = await self._repository.get_all_errors(
            report_date=report_date,
            time_interval=time_interval,
            pagination=pagination,
        )

        return errors

    async def save_report(self, report_create: CaptchaReportCreate) -> None:
        report_information = None
        if report_create.information is not None:
            report_information = report_create.information.model_dump()

        new_report = CaptchaReportInDB(
            status=report_create.status,
            information=report_information,
            uuid=uuid4(),
            report_date=date.today(),
            report_time=datetime.utcnow().strftime("%H:%M:%S"),
        )

        await self._repository.save_report(new_report)
