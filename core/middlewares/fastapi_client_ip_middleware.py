from fastapi import Request
from core.logger import ip_logger


class FastApiClientIpMiddleware:
    async def __call__(self, request: Request, call_next):
        ip_logger.info(f"request from ip {request.client.host}")
        return await call_next(request)
