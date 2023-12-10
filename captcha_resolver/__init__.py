from ultralytics import YOLO

from captcha_resolver.yolov8 import YOLOv8


def init_models():
    """
        Функция для инициализации моделей для распознавания капчи.

        Returns:
            Инициализированные модели для сегментации и детектирования иконок, а также их onnx-версии.

        Description:

        Функция `init_models()` загружает предварительно обученные модели для сегментации и детектирования иконок капчи.
        Функция инициализирует модели в режиме обучения и возвращает их для дальнейшего использования.

        Restriction:

        Функция требует наличия файлов с моделями в формате .pt и .onnx в директории `captcha_resolver/AI_weights`.

        Notes:

        Функция использует библиотеку `torch` для загрузки и инициализации моделей.
    """
    print("Started init models")
    segmentation = YOLO("captcha_resolver/AI_weights/captcha_segmentation_v3.pt")
    detection = YOLO("captcha_resolver/AI_weights/best_v4.pt")
    segmentation.training = False
    detection.training = False
    segmentation_onnx = YOLOv8("captcha_resolver/AI_weights/captcha_segmentation.onnx")
    detection_onnx = YOLOv8("captcha_resolver/AI_weights/best_v3.onnx")
    print("Models inited")
    return segmentation, detection, segmentation_onnx, detection_onnx
