from fastapi import Response, Request
import json
from microservice.data.filters import *


def init_routes(app, service):
    @app.get("/hello")
    async def hello():
        return Response(content=json.dumps({"hello": "world"}),
                        media_type='application/json')

    @app.get("/get_captcha_solve_sequence_old_our")
    async def get_captcha_solve_sequence(request: Request):
        rio = RequestImagesOnly.fromJson(await request.json())
        return Response(content=json.dumps(service.get_captcha_solve_sequence_old(
            request=rio
        )), media_type='application/json')

    @app.get("/get_captcha_solve_sequence_sobel_our")
    async def get_captcha_solve_sequence(request: Request):
        rio = RequestImagesOnly.fromJson(await request.json())
        return Response(content=json.dumps(service.get_captcha_solve_sequence_sobel(
            request=rio)), media_type='application/json')

    @app.get("/get_captcha_solve_sequence_old_business")
    async def get_captcha_solve_sequence_business(request: Request):
        rd = RequestDiscolor.fromJson(await request.json())
        sequence, discolored, captcha, icons, answer = service.get_captcha_solve_sequence_old_business(
            request=rd)
        return Response(content=json.dumps(
            {"coordinate_sequence": sequence, "discolored_captcha": discolored, "captcha": captcha, "icons": icons,
             "answer": answer}),
            media_type='application/json')

    @app.get("/get_captcha_solve_sequence_sobel_business")
    async def get_captcha_solve_sequence_business(request: Request):
        rs = RequestSobel.fromJson(request)
        sequence, discolored, captcha, icons, answer = service.get_captcha_solve_sequence_sobel_business(
            request=rs)
        return Response(content=json.dumps(
            {"coordinate_sequence": sequence, "discolored_captcha": discolored, "captcha": captcha, "icons": icons,
             "answer": answer}), media_type='application/json')
