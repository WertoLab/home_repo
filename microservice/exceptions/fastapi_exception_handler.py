import typing as tp
from pydantic import ValidationError
from fastapi import Request, status
from fastapi.responses import JSONResponse
from microservice.exceptions.api_exception import ApiException
from loguru import logger
from microservice.models.validation_error_message import ValidationErrorMessage


class FastApiExceptionHandler:
    _exceptions: tp.Dict[Exception, tp.Callable[[Exception], None]] = {}

    def __init__(self) -> None:
        self._exceptions = {
            ValidationError: self._handle_validation_error,
            ApiException: self._handle_api_exception,
        }

    def __call__(self, request: Request, exc: Exception):
        exception_handler = self._exceptions.get(type(exc))
        if exception_handler is not None:
            return exception_handler(exc)

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "message": "Internal Server Error!",
            },
        )

    def _handle_validation_error(self, exc: ValidationError) -> None:
        error_messages = [
            ValidationErrorMessage(
                fields=error.get("loc"),
                message=error.get("msg"),
                input_value=error.get("input"),
            ).model_dump_json()
            for error in exc.errors()
        ]

        logger.error(str(error_messages))
        print(str(error_messages))
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status_code": status.HTTP_400_BAD_REQUEST,
                "message": str(error_messages),
            },
        )

    def _handle_api_exception(self, exc: ApiException) -> None:
        logger.error(exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "status_code": exc.status_code,
                "message": exc.message,
            },
        )
