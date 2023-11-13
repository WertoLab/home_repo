import typing as tp

import uuid
import asyncio

from uuid import UUID
from captcha_report.database.repositories import CaptchaReportRepository
from kink import inject
from datetime import date
from captcha_report.models.param_models import ReportPaginationParams
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

    async def get_reports_by_date_and_status(
        self, report_date: date, status: StatusEnum
    ) -> tp.List[CaptchaReportInDB]:
        reports = await self._repository.get_reports_by_date_and_status(
            report_date=report_date,
            status=status,
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

    async def save_report(self, report_create: CaptchaReportCreate) -> None:
        report_information = None
        if report_create.information is not None:
            report_information = report_create.information.model_dump()

        new_report = CaptchaReportInDB(
            status=report_create.status,
            report_date=date.today(),
            uuid=uuid.uuid4(),
            information=report_information,
        )

        await self._repository.save_report(new_report)
