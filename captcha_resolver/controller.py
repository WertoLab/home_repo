import json
from fastapi import Response, Request

from captcha_resolver.data.filters import *


def init_routes(app, service):
    @app.get("/hello")
    async def hello():
        print("HELLO")
        return json.dumps({"hello": "world"})

    @app.get("/delete_captchas")
    async def delete_captchas():
        return json.dumps(service.delete_captchas())

    @app.get("/get_captcha_solve_sequence_segmentation_our")
    async def get_captcha_solve_sequence(request: Request):
        rio = RequestBusiness.fromJson(await request.json())
        return json.dumps(
            service.get_captcha_solve_sequence_segmentation_sobel(request=rio))

    @app.post("/get_captchas1")
    async def get_captcha_solve_sequence_business(request: Request):
        print("Doing captcha1 solve")
        rio = RequestBusiness.fromJson(await request.json())
        sequence = await service.get_captcha_solve_sequence_hybrid_merge_business_async(request=rio)
        # if error:
        #     return Response(content=json.dumps({"status": 0, "request": "ERROR_CAPTCHA_UNSOLVABLE"}),
        #                     media_type="application/json")
        print("Done captcha1 solve")
        return Response(content=json.dumps({"status": 1, "request": sequence}), media_type="application/json")

    @app.route("/get_captchas", methods=["POST"])
    async def get_onnx_check(request: Request):
        print("Doing captcha solve")
        rio = RequestBusiness.fromJson(await request.json())
        sequence, error = service.get_onnx_solver(rio)
        if error:
            return Response(content=json.dumps({"status": 0, "request": "ERROR_CAPTCHA_UNSOLVABLE"}),
                            media_type="application/json")
        print("Done captcha solve")
        return Response(content=json.dumps({"status": 1, "request": sequence}), media_type="application/json")
