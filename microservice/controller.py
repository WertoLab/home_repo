from fastapi import Response, Request
import json

from starlette.responses import FileResponse

from microservice.data.filters import *
from flask import request, jsonify

def init_routes(app, service):
    @app.get("/hello")
    async def hello():
        return Response(content=json.dumps({"hello": "world"}),
                        media_type='application/json')

    @app.get("/get_unresolved_captchas")
    async def get_unresolved_captchas():
        service.get_batch()
        file_path = 'captchas.zip'
        return FileResponse(path=file_path, filename=file_path, media_type='text/mp4')
        # return {"ok": "ok"}

    @app.get("/delete_captchas")
    async def get_unresolved_captchas1():
        return Response(content=json.dumps(service.delete_captchas()),
                        media_type='application/json')

    @app.get("/get_captcha_solve_sequence_segmentation_our")
    async def get_captcha_solve_sequence2(request: Request):
        rio = RequestSobel.fromJson(await request.json())
        return Response(content=json.dumps(service.get_captcha_solve_sequence_segmentation_sobel(
            request=rio
        )), media_type='application/json')


    @app.route("/get_captchas", methods=['GET'])
    async def get_captcha_solve_sequence5():
        rio = RequestBusiness.fromJson(request.get_json())

        sequence, discolored, captcha, icons, answer, error = service.get_captcha_solve_sequence_hybrid_merge_business(request=rio)
        if(error == True):
            return json.dumps({"status": 0, "request": "Not all capchas detected"})
        #return Response(content=json.dumps({"status": 1, "request": sequence}), media_type='application/json')
        return json.dumps({"status": 1, "request": sequence})
