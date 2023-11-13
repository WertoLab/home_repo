from __future__ import annotations
import typing as tp

from pydantic import BaseModel
from captcha_resolver.models.capcha import Capcha


class CapchaRequest(BaseModel):
    body: str
    sobel_filter: int
    imginstructions: str

    method: tp.Any
    coordinatescaptcha: tp.Any
    key: tp.Any
    textinstructions: tp.Any

    def to_capcha(self) -> Capcha:
        return Capcha(
            filter=self.sobel_filter,
            screenshot_captcha=self.body,
            screenshot_icons=self.imginstructions,
        )
