from __future__ import annotations

from pydantic import BaseModel, field_validator
from captcha_resolver.utils.encoding_utils import check_is_base64


class Capcha(BaseModel):
    filter: int
    screenshot_captcha: str
    screenshot_icons: str

    @field_validator("screenshot_captcha", "screenshot_icons", mode="before")
    def check_is_valid_base64_string(cls: Capcha, value: str):
        if not value:
            raise ValueError("Base64 string is empty!")

        if not check_is_base64(value):
            raise ValueError("Invalid base64 string!")

        return value
