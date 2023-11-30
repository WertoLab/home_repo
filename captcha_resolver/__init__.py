import pickle
from ultralytics import YOLO



def init_models():
    segmentation_model = YOLO("captcha_resolver/AI_weights/captcha_segmentation_v3.pt")
    detection_model = YOLO("captcha_resolver/AI_weights/best_v4.pt")

    return segmentation_model, detection_model


segmentation_model: YOLO
detection_model: YOLO

segmentation_model, detection_model = init_models()
