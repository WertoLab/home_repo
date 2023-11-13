import typing as tp
from pydantic import ValidationError
from pydantic_core import ErrorDetails
from captcha_report.models.captcha_report_models import (
    CaptchaReportCreate,
    CaptchaReportInformation,
    StatusEnum,
)


def prepare_resolved_captcha_report() -> CaptchaReportCreate:
    report_create = CaptchaReportCreate(status=StatusEnum.RESOLVED)
    return report_create


def prepare_not_resolved_captcha_report(
    request_body: tp.Dict[str, tp.Any]
) -> CaptchaReportCreate:
    report_create = CaptchaReportCreate(
        status=StatusEnum.NOT_RESOLVED,
        information=CaptchaReportInformation(request_body=request_body),
    )

    return report_create


class CaptchaExceptionReportPreparer:
    _exc: Exception
    _request_body: tp.Dict[str, tp.Any]

    def __init__(self, exc: Exception, request_body: tp.Dict[str, tp.Any]) -> None:
        self._exc = exc
        self._request_body = request_body

    def prepare_exception_report(self) -> CaptchaReportCreate:
        if isinstance(self._exc, ValidationError):
            return self._prepare_validation_exception_report(errors=self._exc.errors())

        return self._prepare_internal_exception_report(errors=str(self._exc))

    def _prepare_validation_exception_report(
        self, errors: tp.List[ErrorDetails]
    ) -> CaptchaReportCreate:
        validation_errors = []
        for error in errors:
            validation_error = {
                "field": error["loc"],
                "message": error["msg"],
                "input_value": error["input"],
            }

            validation_errors.append(validation_error)

        report_information = CaptchaReportInformation(
            request_body=self._request_body,
            errors=validation_errors,
        )

        report_create = CaptchaReportCreate(
            status=StatusEnum.FAILED,
            information=report_information,
        )

        return report_create

    def _prepare_internal_exception_report(self, errors: str) -> CaptchaReportCreate:
        report_information = CaptchaReportInformation(
            request_body=self._request_body,
            errors=errors,
        )

        report_create = CaptchaReportCreate(
            status=StatusEnum.FAILED,
            information=report_information,
        )

        return report_create
