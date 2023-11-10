import uvicorn
import captcha_report
import microservice

from contextlib import asynccontextmanager
from captcha_report.database.connection.alchemy_connection import SqlAlchemyConnection
from microservice.exceptions.fastapi_exception_handler import FastApiExceptionHandler
from fastapi import FastAPI
from loguru import logger
from kink import inject


@inject
@asynccontextmanager
async def lifespan(app: FastAPI, connection: SqlAlchemyConnection):
    async with connection.engine.begin() as conn:
        await conn.run_sync(captcha_report.Base.metadata.create_all)

    yield


app = FastAPI(lifespan=lifespan)
app.add_exception_handler(Exception, FastApiExceptionHandler())
app.include_router(captcha_report.router)
app.include_router(microservice.router)


logger.remove()
logger.add(
    # "logs/error_{time}.log", # prod
    "logs/error.log",
    format="{time} | {level} | {message}",
    level="ERROR",
    rotation="1 day",
    compression="zip",
)


if __name__ == "__main__":
    uvicorn.run(
        "async_app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
