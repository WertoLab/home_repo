from sqlalchemy import Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import func
from captcha_report.database.models.base import Base
from datetime import date
from captcha_report.models.captcha_report_models import StatusEnum, CaptchaReportInDB


class CaptchaReport(Base):
    __tablename__ = "captcha_report"

    report_date: Mapped[date] = mapped_column(nullable=False, default=func.current_date)
    status: Mapped[StatusEnum] = mapped_column(nullable=False)
    information: Mapped[str] = mapped_column(Text, nullable=True)

    def __str__(self) -> str:
        return f"{self.uuid} | {self.status} | {self.resolve_datetime}"

    def to_domain_model(self) -> CaptchaReportInDB:
        return CaptchaReportInDB(
            uuid=self.uuid,
            report_date=self.report_date,
            information=self.information,
            status=self.status,
        )
