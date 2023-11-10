import typing as tp
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
    AsyncEngine,
    AsyncSession,
)


class SqlAlchemyConnection:
    _engine: AsyncEngine
    _session_factory: async_sessionmaker[AsyncSession]

    def __init__(self, database_url: str, echo: bool = False) -> None:
        self._engine = create_async_engine(url=database_url, echo=echo)
        self._session_factory = async_sessionmaker(
            self._engine,
            autocommit=False,
            expire_on_commit=False,
        )

    async def get_session(self) -> AsyncSession:
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()

            except SQLAlchemyError as exc:
                await session.rollback()
                raise

            finally:
                await session.close()

    @property
    def engine(self) -> AsyncEngine:
        return self._engine
