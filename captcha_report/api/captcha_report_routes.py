from uuid import UUID
from fastapi import APIRouter, Depends
from kink import di

from captcha_report.services.captcha_report_service import CaptchaReportService
from captcha_report.models.response_models import (
    CaptchaReportListResponse,
    CaptchaReportListResponseWithCount,
)
from captcha_report.models.param_models import (
    ReportGetParams,
    StatisticDatetimeParams,
    ErrorsGetParams,
)

router = APIRouter(prefix="/reports")


@router.get("/filter", response_model=CaptchaReportListResponseWithCount)
async def get_reports_by_date_and_status(
    params: ReportGetParams = Depends(),
    service: CaptchaReportService = Depends(lambda: di[CaptchaReportService]),
):
    if params.limit is not None:
        count, reports = await service.get_reports_by_date_and_status_with_count(
            report_date=params.iso_date,
            status=params.status,
            pagination=params.get_pagination(),
        )

        return {
            "total_count": count,
            "reports": reports,
        }

    reports = await service.get_reports_by_date_and_status(
        report_date=params.iso_date,
        status=params.status,
    )

    return {
        "total_count": len(reports),
        "reports": reports,
    }


@router.get("/filter/{uuid:str}")
async def get_report_by_uuid(
    uuid: UUID,
    service: CaptchaReportService = Depends(lambda: di[CaptchaReportService]),
):
    report = await service.get_report_by_uuid(uuid)
    return report


@router.get("/statistic/")
async def count_reports_statistic_by_date(
    params: StatisticDatetimeParams = Depends(),
    service: CaptchaReportService = Depends(lambda: di[CaptchaReportService]),
):
    if params.utc_time_interval is not None:
        statistic = await service.count_reports_statistic_by_datetime(
            report_date=params.iso_date,
            time_interval=params.get_time_interval(),
        )

        return statistic

    statistic = await service.count_reports_statistic_by_date(params.iso_date)
    return statistic


@router.get("/errors")
async def get_all_errors(
    params: ErrorsGetParams = Depends(),
    service: CaptchaReportService = Depends(lambda: di[CaptchaReportService]),
):
    errors = await service.get_all_errors(
        report_date=params.iso_date,
        time_interval=params.get_time_interval(),
        pagination=params.get_pagination(),
    )

    return errors


@router.get("/failed", response_model=CaptchaReportListResponse)
async def get_failed_reports(
    params: ErrorsGetParams = Depends(),
    service: CaptchaReportService = Depends(lambda: di[CaptchaReportService]),
):
    failed_reports = await service.get_failed_reports_by_datetime_and_status(
        report_date=params.iso_date,
        time_interval=params.get_time_interval(),
        pagination=params.get_pagination(),
    )

    return {
        "reports": failed_reports,
    }
