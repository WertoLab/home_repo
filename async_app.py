import uvicorn
import captcha_report
import microservice

from contextlib import asynccontextmanager
from captcha_report.database.utils.create_tables import create_captcha_report_tables
from microservice.exceptions.fastapi_exception_handler import FastApiExceptionHandler
from fastapi import FastAPI
from kink import inject


@inject
@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_captcha_report_tables()
    yield


app = FastAPI(lifespan=lifespan)
app.add_exception_handler(Exception, FastApiExceptionHandler())
app.include_router(captcha_report.router)
app.include_router(microservice.router)


if __name__ == "__main__":
    uvicorn.run(
        "async_app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
