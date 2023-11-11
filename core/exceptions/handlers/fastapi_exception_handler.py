import typing as tp
from fastapi import Request, status
from fastapi.responses import JSONResponse
from core.exceptions.handlers.abstractions.exception_handler import ExceptionHandler
from core.logger import internal_error_logger


class FastApiExceptionHandler:
    _handlers: tp.List[ExceptionHandler]

    def __init__(self) -> None:
        self._handlers = []

    async def __call__(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:
            for handler in self._handlers:
                response = await handler.handle(exc)
                if response is not None:
                    return JSONResponse(
                        status_code=response.status_code,
                        content={
                            "status_code": response.status_code,
                            "message": response.message,
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

    def add_exception_handler(self, handler: ExceptionHandler) -> None:
        self._handlers.append(handler)
