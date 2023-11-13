import typing as tp
from sqlalchemy import JSON, TIME
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import func
from captcha_report.database.models.base import Base
from datetime import date, time
from captcha_report.models.captcha_report_models import StatusEnum, CaptchaReportInDB


class CaptchaReport(Base):
    __tablename__ = "captcha_report"

    report_date: Mapped[date] = mapped_column(nullable=False, default=func.current_date)
    report_time: Mapped[time] = mapped_column(
        TIME(timezone=False),
        nullable=False,
        default=func.current_time,
    )

    status: Mapped[StatusEnum] = mapped_column(nullable=False)
    information: Mapped[tp.Dict[str, tp.Any]] = mapped_column(
        JSON(none_as_null=True),
        nullable=True,
    )

    def __str__(self) -> str:
        return f"{self.uuid} | {self.status} | {self.report_date} | {self.report_time}"

    def to_domain_model(self) -> CaptchaReportInDB:
        return CaptchaReportInDB(
            uuid=self.uuid,
            report_date=self.report_date,
            report_time=self.report_time,
            information=self.information,
            status=self.status,
        )
