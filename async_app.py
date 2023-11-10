import asyncio
import uvicorn
import functools
from ultralytics import YOLO
from microservice.exceptions.fastapi_exception_handler import FastApiExceptionHandler
from microservice.service import Service
from fastapi import FastAPI
from microservice.models.capcha_request import CapchaRequest
from concurrent.futures import ThreadPoolExecutor
from loguru import logger


app = FastAPI()
app.add_exception_handler(Exception, FastApiExceptionHandler())


service = Service()
logger.remove()
logger.add(
    # "logs/error_{time}.log", # prod
    "logs/error.log",
    format="{time} | {level} | {message}",
    level="ERROR",
    rotation="1 day",
    compression="zip",
)


def init_models():
    segmentation_model = YOLO("microservice/AI_weights/captcha_segmentation_v2.pt")
    detection_model = YOLO("microservice/AI_weights/best_v3.pt")
    return segmentation_model, detection_model


segmentation_model: YOLO
detection_model: YOLO

segmentation_model, detection_model = init_models()


@app.post("/get_captchas")
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


if __name__ == "__main__":
    uvicorn.run(
        "async_app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
