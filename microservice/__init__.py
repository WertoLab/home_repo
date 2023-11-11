import pickle
from ultralytics import YOLO
from .async_controller import router
from microservice.utils.logger import internal_error_logger, validation_error_logger


def init_models():
    label_encoder = pickle.load(open("microservice/label_encoder.pkl", "rb"))
    segmentation_model = YOLO("microservice/AI_weights/captcha_segmentation_v2.pt")
    detection_model = YOLO("microservice/AI_weights/best_v3.pt")

    return segmentation_model, detection_model, label_encoder


segmentation_model: YOLO
detection_model: YOLO

segmentation_model, detection_model, labal_encoder = init_models()
