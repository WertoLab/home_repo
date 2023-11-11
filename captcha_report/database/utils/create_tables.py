from core.config import settings
from captcha_report import Base
from captcha_report.database.connection import SqlAlchemyConnection


async def create_captcha_report_tables() -> None:
    connection = SqlAlchemyConnection(
        database_url=settings.database_url,
        echo=settings.echo,
    )

    async with connection.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
