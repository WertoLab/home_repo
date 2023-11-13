from __future__ import annotations
import json
import typing as tp

from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
    AsyncEngine,
    AsyncSession,
)


class SqlAlchemyConnection:
    _instance: SqlAlchemyConnection = None

    _engine: AsyncEngine
    _session_factory: async_sessionmaker[AsyncSession]

    def __new__(cls, *args: tp.Any, **kwargs: tp.Any):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self, database_url: str, echo: bool = False) -> None:
        self._engine = create_async_engine(
            url=database_url,
            echo=echo,
            json_serializer=lambda data: json.dumps(data, ensure_ascii=False),
            json_deserializer=lambda data: json.loads(data),
        )

        self._session_factory = async_sessionmaker(
            self._engine,
            autocommit=False,
            expire_on_commit=False,
        )

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        return self._session_factory

    @property
    def engine(self) -> AsyncEngine:
        return self._engine
