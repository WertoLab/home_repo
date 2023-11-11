import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
from fastapi import APIRouter, Depends
from microservice.models.capcha_request import CapchaRequest
from microservice.service import Service
from captcha_report.models.captcha_report_models import CaptchaReportCreate, StatusEnum
from captcha_report.services.captcha_report_service import CaptchaReportService
from kink import di


router = APIRouter(prefix="")
service = Service()


@router.post("/get_captchas")
async def get_captcha_solve_sequence_business(
    request: CapchaRequest,
    report_service: CaptchaReportService = Depends(lambda: di[CaptchaReportService]),
):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=5) as pool:
        sequence, error = await loop.run_in_executor(
            executor=pool,
            func=functools.partial(
                service.get_captcha_solve_sequence_hybrid_merge_business,
                request=request.to_capcha(),
            ),
        )

    if error:
        report_create = CaptchaReportCreate(status=StatusEnum.NOT_RESOLVED)
        await report_service.save_report(report_create)

        return {
            "status": 0,
            "request": "ERROR_CAPTCHA_UNSOLVABLE",
        }

    report_create = CaptchaReportCreate(status=StatusEnum.RESOLVED)
    await report_service.save_report(report_create)

    return {
        "status": 1,
        "request": sequence,
    }
