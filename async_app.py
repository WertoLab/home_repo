import uvicorn
import captcha_report
import captcha_resolver

from contextlib import asynccontextmanager
from core.exceptions.handlers.fastapi_exception_handler import FastApiExceptionHandler
from core.middlewares.fastapi_client_ip_middleware import FastApiClientIpMiddleware
from captcha_report.database.utils.create_tables import create_captcha_report_tables
from fastapi import FastAPI
from kink import inject


@inject
@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_captcha_report_tables()
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(captcha_report.router)
app.include_router(captcha_resolver.router)

app.middleware("http")(FastApiClientIpMiddleware())
app.middleware("http")(FastApiExceptionHandler())


if __name__ == "__main__":
    uvicorn.run(
        "async_app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
