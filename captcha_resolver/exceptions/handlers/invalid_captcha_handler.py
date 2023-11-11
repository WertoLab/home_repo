import typing as tp

from kink import di
from fastapi import status

from captcha_report.models.captcha_report_models import CaptchaReportCreate, StatusEnum
from core.exceptions.models.exception_response import ExceptionResponse
from core.logger import validation_error_logger

from core.exceptions.handlers.abstractions.exception_handler import ExceptionHandler
from captcha_report.services.captcha_report_service import CaptchaReportService
from captcha_resolver.exceptions import InvalidCaptchaException


class InvalidCaptchaHandler(ExceptionHandler):
    _report_service: CaptchaReportService

    def __init__(self, report_service=di[CaptchaReportService]):
        self._report_service = report_service

    async def handle(self, exc: Exception) -> None:
        if not isinstance(exc, InvalidCaptchaException):
            return

        validation_error_logger.error(exc.to_json())
        new_report = CaptchaReportCreate(
            information=exc.to_json(),
            status=StatusEnum.FAILED,
        )

        await self._report_service.save_report(new_report)
        return ExceptionResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            message=exc.error_message,
        )
