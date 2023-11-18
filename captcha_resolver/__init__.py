import pickle
from ultralytics import YOLO
from .async_controller import router


def init_models():
    filename = "captcha_resolver/label_encoder.pkl"
    with open(filename, "rb") as f:
        label_encoder = pickle.load(f)

    segmentation_model = YOLO("captcha_resolver/AI_weights/captcha_segmentation_v3.pt")
    detection_model = YOLO("captcha_resolver/AI_weights/best_v4.pt")

    return segmentation_model, detection_model, label_encoder


segmentation_model: YOLO
detection_model: YOLO

try:
    segmentation_model, detection_model, labal_encoder = init_models()
except Exception:
    pass
