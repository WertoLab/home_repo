from fastapi import Response, Request
import json
from microservice.data.filters import *
from flask import request, jsonify


def init_routes(app, service):
    @app.get("/hello")
    async def hello():
        return Response(content=json.dumps({"hello": "world"}),
                        media_type='application/json')

    @app.get("/get_captcha_solve_sequence_hybrid_merge")
    async def get_captcha_solve_sequence4(request: Request):
        rio = RequestSobel.fromJson(await request.json())
        sequence, discolored, captcha, icons, answer = service.get_captcha_solve_sequence_hybrid_merge(request=rio)
        return Response(content=json.dumps({"status": 1, "request": sequence}), media_type='application/json')

    @app.route("/get_captcha_solve_sequence_hybrid_merge_business", methods=['GET'])
    async def get_captcha_solve_sequence5():
        rio = RequestBusiness.fromJson(request.get_json())
        if (service.get_captcha_solve_sequence_hybrid_merge_business(request=rio) == True):
            return json.dumps({"status": 0, "error": "Not detected whole captcha"})
        sequence, discolored, captcha, icons, answer = service.get_captcha_solve_sequence_hybrid_merge_business(
            request=rio)
        # return Response(content=json.dumps({"status": 1, "request": sequence}), media_type='application/json')
        return json.dumps({"status": 1, "request": sequence})
