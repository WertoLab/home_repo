import pickle
from ultralytics import YOLO
from .async_controller import router


def init_models():
    label_encoder = pickle.load(open("captcha_resolver/label_encoder.pkl", "rb"))
    segmentation_model = YOLO("captcha_resolver/AI_weights/captcha_segmentation_v2.pt")
    detection_model = YOLO("captcha_resolver/AI_weights/best_v3.pt")

    return segmentation_model, detection_model, label_encoder


segmentation_model: YOLO
detection_model: YOLO

segmentation_model, detection_model, labal_encoder = init_models()
