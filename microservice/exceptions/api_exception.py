from __future__ import annotations
import typing as tp


class ApiException(Exception):
    status_code: int
    message: str

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message

    @classmethod
    def bad_request(cls: tp.Type[ApiException], message: str) -> None:
        return cls(status_code=400, message=message)
