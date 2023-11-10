import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
from fastapi import APIRouter
from microservice.models.capcha_request import CapchaRequest
from microservice.service import Service


router = APIRouter(prefix="")
service = Service()


@router.post("/get_captchas")
async def get_captcha_solve_sequence_business(request: CapchaRequest):
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
        return {
            "status": 0,
            "request": "ERROR_CAPTCHA_UNSOLVABLE",
        }

    return {
        "status": 1,
        "request": sequence,
    }
