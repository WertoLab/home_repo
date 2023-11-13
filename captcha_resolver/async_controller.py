import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from kink import di
from fastapi import APIRouter, Depends, Request

from captcha_report.services.captcha_report_service import CaptchaReportService
from captcha_resolver.models.capcha_request import CapchaRequest
from captcha_resolver.service import Service
from captcha_resolver.utils.report_utils import (
    CaptchaExceptionReportPreparer,
    prepare_resolved_captcha_report,
    prepare_not_resolved_captcha_report,
)


router = APIRouter(prefix="")
service = Service()


@router.post("/get_captchas")
async def get_captcha_solve_sequence_business(
    request: Request,
    report_service: CaptchaReportService = Depends(lambda: di[CaptchaReportService]),
):
    try:
        request_body = await request.json()
        request = CapchaRequest(**request_body)

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
            report = prepare_not_resolved_captcha_report(request_body)
            await report_service.save_report(report)

            return {
                "status": 0,
                "request": "ERROR_CAPTCHA_UNSOLVABLE",
            }

        report = prepare_resolved_captcha_report()
        await report_service.save_report(report)

        return {
            "status": 1,
            "request": sequence,
        }

    except Exception as exc:
        report_preparer = CaptchaExceptionReportPreparer(exc, request_body)
        exception_report = report_preparer.prepare_exception_report()
        await report_service.save_report(exception_report)
        raise exc
