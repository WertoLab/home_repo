import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
from fastapi import Request
import json

from ultralytics import YOLO
from captcha_resolver.data.filters import *
from fastapi import APIRouter, Response
from captcha_resolver.yolov8 import YOLOv8


def init_models():
    segmentation_model = YOLO("captcha_resolver/AI_weights/captcha_segmentation_v2.pt")
    detection_model = YOLO("captcha_resolver/AI_weights/best_v3.pt")
    segmentation_onnx_model = YOLOv8("captcha_resolver/AI_weights/captcha_segmentation.onnx")
    detection_onnx_model = YOLOv8("captcha_resolver/AI_weights/best_v3.onnx")
    return segmentation_model, detection_model, segmentation_onnx_model, detection_onnx_model


segmentation_model: YOLO
detection_model: YOLO
segmentation_onnx_model: YOLOv8
detection_onnx_model: YOLOv8

segmentation_model, detection_model, segmentation_onnx_model, detection_onnx_model = init_models()


def init_routes(app, service):

    @app.get("/hello")
    async def hello():
        return json.dumps({"hello": "world"})

    @app.get("/get_unresolved_captchas")
    async def get_unresolved_captchas():
        service.get_batch()
        file_path = "captchas.zip"
        return send_file(file_path, as_attachment=True)
        # return {"ok": "ok"}

    @app.get("/delete_captchas")
    async def delete_captchas():
        return json.dumps(service.delete_captchas())

    @app.get("/get_captcha_solve_sequence_segmentation_our")
    async def get_captcha_solve_sequence(request: Request):
        rio = RequestBusiness.fromJson(await request.json())
        return json.dumps(
            service.get_captcha_solve_sequence_segmentation_sobel(request=rio)
        )

    @app.post("/get_captchas1")
    async def get_captcha_solve_sequence_business(request: Request):
        rio = RequestBusiness.fromJson(await request.json())
        # print(request.environ)
        # f = open(str(request.environ)+".txt","w+")
        sequence, error = service.get_captcha_solve_sequence_hybrid_merge_business(
            request=rio)
        if error:
            return Response(content=json.dumps({"status": 0, "request": "ERROR_CAPTCHA_UNSOLVABLE"}), media_type="application/json")#json.dumps({"status": 0, "request": "ERROR_CAPTCHA_UNSOLVABLE"})
        # return Response(content=json.dumps({"status": 1, "request": sequence}), media_type='application/json')
        return Response(content=json.dumps({"status": 1, "request": sequence}), media_type="application/json")

    @app.route("/get_captchas", methods=["POST"])
    async def get_onnx_check(request: Request):
        rio = RequestBusiness.fromJson(await request.json())
        sequence, error = service.get_onnx_solver(rio)
        if error:
            return Response(content=json.dumps({"status": 0, "request": "ERROR_CAPTCHA_UNSOLVABLE"}),
                            media_type="application/json")  # json.dumps({"status": 0, "request": "ERROR_CAPTCHA_UNSOLVABLE"})
        # return Response(content=json.dumps({"status": 1, "request": sequence}), media_type='application/json')
        return Response(content=json.dumps({"status": 1, "request": sequence}), media_type="application/json")
