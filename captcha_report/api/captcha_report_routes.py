from fastapi import APIRouter, Depends
from kink import di
from captcha_report.services.captcha_report_service import CaptchaReportService
from captcha_report.models.captcha_report_models import StatusEnum
from captcha_report.models.response_models import CaptchaReportResponse
from datetime import date

router = APIRouter(prefix="/reports")


@router.get("/statistic/")
async def count_reports_statistic_by_date(
    iso_date: date,
    service: CaptchaReportService = Depends(lambda: di[CaptchaReportService]),
):
    statistic = await service.count_reports_statistic_by_date(report_date=iso_date)
    return statistic


@router.get("/filter", response_model=CaptchaReportResponse)
async def get_reports_by_date_and_status(
    status: StatusEnum,
    iso_date: date,
    service: CaptchaReportService = Depends(lambda: di[CaptchaReportService]),
):
    reports = await service.get_reports_by_date_and_status(
        report_date=iso_date,
        status=status,
    )

    return {
        "report_count": len(reports),
        "reports": reports,
    }
