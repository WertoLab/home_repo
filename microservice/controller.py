from fastapi import Request
import json

from ultralytics import YOLO

from microservice.data.filters import *
from flask import request, send_file
import torch
from microservice.AI_models.ClassificationModel import AlexNet
import pickle

def init_models():
    segmentation_model = YOLO("microservice/AI_weights/captcha_segmentation.pt")
    detection_model = YOLO("microservice/AI_weights/best_v3.pt")
    alexnet = AlexNet()
    alexnet.load_state_dict(torch.load("microservice/AI_weights/smartsolver_weights_1_6.pth", map_location='cpu'))
    alexnet.eval()
    label_encoder = pickle.load(open("microservice/label_encoder.pkl", 'rb'))
    return segmentation_model, detection_model, alexnet, label_encoder


segmentation_model: YOLO
detection_model: YOLO
alexnet = 0
label_encoder = 0

segmentation_model, detection_model, alexnet, label_encoder = init_models()


def init_routes(app, service):
    @app.get("/hello")
    async def hello():
        return json.dumps({"hello": "world"})

    @app.get("/get_unresolved_captchas")
    async def get_unresolved_captchas():
        service.get_batch()
        file_path = 'captchas.zip'
        return send_file(file_path, as_attachment=True)
        # return {"ok": "ok"}

    @app.get("/delete_captchas")
    async def delete_captchas():
        return json.dumps(service.delete_captchas())

    @app.get("/get_captcha_solve_sequence_segmentation_our")
    async def get_captcha_solve_sequence(request: Request):
        rio = RequestBusiness.fromJson(await request.json())
        return json.dumps(service.get_captcha_solve_sequence_segmentation_sobel(request=rio))

    @app.route("/get_captchas", methods=['POST'])
    async def get_captcha_solve_sequence_business():
        rio = RequestBusiness.fromJson(request.get_json())

        sequence, discolored, captcha, icons, answer, error = service.get_captcha_solve_sequence_hybrid_merge_business(
            request=rio)
        if error:
            return json.dumps({"status": 0, "request": "ERROR_CAPTCHA_UNSOLVABLE"})
        # return Response(content=json.dumps({"status": 1, "request": sequence}), media_type='application/json')
        return json.dumps({"status": 1, "request": sequence})
