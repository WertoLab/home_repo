import uvicorn
import captcha_report
import captcha_resolver

from captcha_resolver import InvalidCaptchaHandler
from contextlib import asynccontextmanager
from core.exceptions.handlers.fastapi_exception_handler import FastApiExceptionHandler
from captcha_report.database.utils.create_tables import create_captcha_report_tables
from fastapi import FastAPI
from kink import di, inject


@inject
@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_captcha_report_tables()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(captcha_report.router)
app.include_router(captcha_resolver.router)

exception_handler = FastApiExceptionHandler()
exception_handler.add_exception_handler(InvalidCaptchaHandler())
app.middleware("http")(exception_handler)


if __name__ == "__main__":
    uvicorn.run(
        "async_app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
