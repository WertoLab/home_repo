from fastapi import FastAPI, Request
from typing import List
from fastapi.responses import JSONResponse
from fastapi import Response
import json
from starlette.middleware.cors import CORSMiddleware

from service import Service

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']
)
service = Service()


@app.get("/get_captcha_solve_sequence_old_our")
async def get_captcha_solve_sequence(info: Request):
    return Response(content=json.dumps(service.get_captcha_solve_sequence_old(await info.json())),
                    media_type='application/json')


@app.get("/get_captcha_solve_sequence_sobel_our")
async def get_captcha_solve_sequence(info: Request):
    return Response(content=json.dumps(service.get_captcha_solve_sequence_sobel(await info.json())),
                    media_type='application/json')


@app.get("/get_captcha_solve_sequence_old_business")
async def get_captcha_solve_sequence_business(info: Request):
    sequence, discolored, captcha, icons, answer = service.get_captcha_solve_sequence_old_business(await info.json())
    return Response(content=json.dumps({"coordinate_sequence": sequence, "discolored_captcha": discolored, "captcha": captcha, "icons": icons, "answer": answer}),
                    media_type='application/json')


@app.get("/get_captcha_solve_sequence_sobel_business")
async def get_captcha_solve_sequence_business(info: Request):
    sequence, discolored, captcha, icons, answer = service.get_captcha_solve_sequence_sobel_business(await info.json())
    return Response(content=json.dumps({"coordinate_sequence": sequence, "discolored_captcha": discolored, "captcha": captcha, "icons": icons, "answer": answer}),
                    media_type='application/json')
