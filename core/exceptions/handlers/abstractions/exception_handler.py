import typing as tp
from abc import abstractmethod
from core.exceptions.models.exception_response import ExceptionResponse


class ExceptionHandler(tp.Protocol):
    @abstractmethod
    async def handle(self, exc: Exception) -> tp.Optional[ExceptionResponse]:
        ...
