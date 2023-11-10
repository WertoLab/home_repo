from sqlalchemy import Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import func
from captcha_report.database.models.base import Base
from datetime import datetime
from captcha_report.models.captcha_report import StatusEnum


class CaptchaReport(Base):
    __tablename__ = "captcha_report"

    resolve_datetime: Mapped[datetime] = mapped_column(nullable=False, default=func.now)
    status: Mapped[StatusEnum] = mapped_column(nullable=False)
    information: Mapped[str] = mapped_column(Text, nullable=True)

    def __str__(self) -> str:
        return f"{self.uuid} | {self.status} | {self.resolve_datetime}"
