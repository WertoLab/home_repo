from ultralytics import YOLO
from .async_controller import router


def init_models():
    segmentation_model = YOLO("microservice/AI_weights/captcha_segmentation_v2.pt")
    detection_model = YOLO("microservice/AI_weights/best_v3.pt")
    return segmentation_model, detection_model


segmentation_model: YOLO
detection_model: YOLO

segmentation_model, detection_model = init_models()
