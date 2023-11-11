from __future__ import annotations
import typing as tp

from pydantic import BaseModel, model_validator
from captcha_resolver.exceptions import InvalidCaptchaException
from captcha_resolver.models.capcha import Capcha


class CapchaRequest(BaseModel):
    method: str
    coordinatescaptcha: int
    key: str
    body: str
    imginstructions: str
    textinstructions: str
    sobel_filter: int

    @model_validator(mode="before")
    def check_fields_is_not_null(data: tp.Any):
        if any(
            (
                "body" not in data,
                "sobel_filter" not in data,
                "imginstructions" not in data,
            )
        ):
            raise InvalidCaptchaException(
                error_message="Fields body, sobel_label and imginstructions is required, but got empty value!",
                input_value="null",
            )

        return data

    def to_capcha(self) -> Capcha:
        return Capcha(
            filter=self.sobel_filter,
            screenshot_captcha=self.body,
            screenshot_icons=self.imginstructions,
        )
