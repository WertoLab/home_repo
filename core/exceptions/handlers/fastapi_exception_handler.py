import typing as tp
import json

from fastapi import Request, status
from fastapi.responses import JSONResponse
from core.logger import internal_error_logger, validation_error_logger
from pydantic import ValidationError
from core.exceptions.api_exception import ApiException


class FastApiExceptionHandler:
    async def __call__(self, request: Request, call_next):
        # return await call_next(request)

        try:
            return await call_next(request)
        except Exception as exc:
            return self._get_exception_response(exc)

    def _get_exception_response(self, exc: Exception):
        if isinstance(exc, ApiException):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "status_code": exc.status_code,
                    "message": exc.message,
                },
            )

        if isinstance(exc, ValidationError):
            validation_errors = []
            for error in exc.errors():
                validation_error = {
                    "field": error["loc"],
                    "message": error["msg"],
                    "input_value": error["input"],
                }

                validation_errors.append(validation_error)

            validation_error_logger.error(json.dumps(validation_errors))
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
                    "message": validation_errors,
                },
            )

        internal_error_logger.error(str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "message": "Internal Server Error!",
            },
        )
