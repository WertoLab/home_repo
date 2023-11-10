from fastapi import APIRouter, Depends
from captcha_report.services.captcha_report_service import CaptchaReportService

router = APIRouter(prefix="/reports")
