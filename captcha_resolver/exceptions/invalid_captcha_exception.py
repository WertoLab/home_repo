from __future__ import annotations
import typing as tp
import json


class InvalidCaptchaException(Exception):
    error_message: str
    input_value: str

    def __init__(self, error_message: str, input_value: str) -> None:
        self.error_message = error_message
        self.input_value = input_value

    def to_json(self) -> str:
        data = {
            "error_message": self.error_message,
            "input_value": self.input_value,
        }

        return json.dumps(data)
