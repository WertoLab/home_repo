from __future__ import annotations

from pydantic import BaseModel
from microservice.models.capcha import Capcha


class CapchaRequest(BaseModel):
    method: str
    coordinatescaptcha: int
    key: str
    body: str
    imginstructions: str
    textinstructions: str
    sobel_filter: int

    def to_capcha(self) -> Capcha:
        return Capcha(
            filter=self.sobel_filter,
            screenshot_captcha=self.body,
            screenshot_icons=self.imginstructions,
        )
