import os
import boto3
import base64
import shutil
import numpy as np

import cv2
import torch
import asyncio
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ThreadPoolExecutor

import Config

from captcha_resolver.data.filters import RequestBusiness
from captcha_resolver.preprocess import preprocess_captcha_sobel
from captcha_resolver.AI_models.ClassificationModel import AlexNet

from ultralytics import YOLO

from captcha_resolver import init_models
from captcha_resolver.yolov8 import YOLOv8

segmentation_model: YOLO
detection_model: YOLO
segmentation_onnx_model: YOLOv8
detection_onnx_model: YOLOv8
segmentation_model, detection_model, segmentation_onnx_model, detection_onnx_model = init_models()


def readb64(encoded_data):
    """
        Функция для декодирования изображения в формате Base64 и преобразования его в формат NumPy.

        Args:
            encoded_data: Изображение в формате Base64.

        Returns:
            Изображение в формате NumPy.
    """
    nparr = np.frombuffer(encoded_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return img


def b64_decode(im_b64: str):
    """
        Декодирует изображение, закодированное в формате base64.

        Args:
            im_b64: Закодированное в формате base64 изображение в виде строки.

        Returns:
            Декодированное изображение в виде массива NumPy.
    """
    img_bytes = base64.b64decode(im_b64.encode("utf-8"))
    img = readb64(img_bytes)
    img_arr = np.asarray(img)
    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    return img_bgr


class Service:
    your_classes = [
        "arrow",
        "book",
        "bucket",
        "clock",
        "cloud",
        "compass",
        "electro",
        "eye",
        "face",
        "factory",
        "fire",
        "flag",
        "hand",
        "heart",
        "house",
        "key",
        "keyboard",
        "light",
        "lightning",
        "lock",
        "magnifier",
        "mail",
        "microphone",
        "monitor",
        "paper",
        "paperclip",
        "pen",
        "person",
        "photo",
        "pill",
        "scissors",
        "shop_cart",
        "sound",
        "star",
        "store_cart",
        "t-shirt",
        "ticket",
        "traffic_light",
        "umbrella",
        "water",
        "wrench",
    ]

    def get_coordinates_onnx(self, name, boxes, class_ids):
        """
            Извлекает координаты центра объекта с заданным именем из массива bounding boxes и class_ids.

            Args:
                name: Имя объекта, для которого необходимо получить координаты центра.
                boxes: Массив bounding boxes в формате [N, (x_min, y_min, x_max, y_max)]
                class_ids: Массив class_ids в формате [N].

            Returns:
                Координаты центра объекта с заданным именем, если такой объект найден, иначе None, None.
        """
        for i in range(len(class_ids)):
            if self.your_classes[class_ids[i]] == name:
                return (int(boxes[i][2]) + int(boxes[i][0])) / 2, (int(boxes[i][3]) + int(boxes[i][1])) / 2

        return None, None

    def get_onnx_inference(self, captcha, icons, model):
        """
            Выполняет распознавание капчи на основе модели YOLOv8, возвращает последовательность координат объектов.

            Args:
                captcha: Изображение капчи в формате NumPy.
                icons: Список объектов, которые необходимо распознать.
                model: Модель YOLOv8, загруженная в память.

            Returns:
                Последовательность координат объектов, если все объекты были распознаны, иначе None.
        """
        sequence = []
        index = 1
        yolov8 = model
        boxes, scores, class_ids = yolov8(captcha)
        for icon in icons:
            name = self.classify_image(icon)
            x, y = self.get_coordinates_onnx(name, boxes, class_ids)
            sequence.append({"x": x, "y": y})
            index += 1
        return sequence

    def get_onnx_solver(self, data):
        """
            Решает капчу на основе двух моделей YOLOv8: для обнаружения объектов и для их сегментации.

            Args:
                data: Объект, содержащий скриншоты капчи и иконок.

            Returns:
                Последовательность объектов, извлеченных из капчи.
        """
        captcha = b64_decode(data.screenshot_captcha)
        icons = preprocess_captcha_sobel(icons=b64_decode(data.screenshot_icons))
        detections = self.get_onnx_inference(captcha, icons, detection_onnx_model)
        segmentations = self.get_onnx_inference(captcha, icons, segmentation_onnx_model)
        return self.merge(detections, segmentations)

    def get_boxes(self, result):
        """
            Извлекает ограничивающие рамки объектов из результатов обнаружения капчи.

            Args:
                result: Результаты обнаружения капчи в формате PyTorch Tensor.

            Returns:
                Массив ограничивающих рамок в формате [[x_up, y_up, x_bottom, y_bottom], ...].
        """
        boxes = []
        all_params = result[0].boxes
        for i in range(len(result[0].boxes.conf.cpu())):
            if np.array(all_params.conf.cpu()[i]) > 0.05:
                x_up, y_up = (all_params.xyxy.cpu()[i][0].numpy(),
                              all_params.xyxy.cpu()[i][1].numpy(),)
                x_bottom, y_bottom = (all_params.xyxy.cpu()[i][2].numpy(),
                                      all_params.xyxy.cpu()[i][3].numpy(),)
                boxes.append([x_up, y_up, x_bottom, y_bottom])
        return boxes

    def sobel_filter(self, threshold, img):
        """
            Применяет фильтр Собеля к изображению, чтобы выделить края.

            Args:
                threshold: Значение порога для определения значащих краев.
                img: Исходное изображение в формате NumPy.

            Returns:
                Обработанное изображение с выделенными краями.
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        G_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        G_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        rows, columns = img.shape
        mag = np.zeros(img.shape, dtype=np.float32)
        for i in range(0, rows - 2):
            for j in range(0, columns - 2):
                v = np.sum(np.sum(G_x * img[i: i + 3, j: j + 3]))
                h = np.sum(np.sum(G_y * img[i: i + 3, j: j + 3]))
                mag[i + 1, j + 1] = np.sqrt((v ** 2) + (h ** 2))
                if mag[i + 1, j + 1] < threshold:
                    mag[i + 1, j + 1] = 0
        processed_image = mag.astype(np.uint8)
        return cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)

    def get_boxes_detection(self, name, prediction, model):
        """
            Извлекает координаты центра объекта с заданным именем из результатов обнаружения капчи.

            Args:
                name: Имя объекта, для которого необходимо получить координаты.
                prediction: Результат обнаружения капчи в формате PyTorch Tensor.
                model: Модель YOLOv8, используемая для обнаружения объекта.

            Returns:
                Координаты центра объекта с заданным именем, если такой объект найден, иначе None, None.
        """
        for index, box in enumerate(self.get_boxes(prediction)):
            if model.model.names[int(prediction[0].boxes.cls.cpu()[index])] == name:
                # print(box)
                return (int(box[2]) + int(box[0])) / 2, (int(box[3]) + int(box[1])) / 2
        return None, None

    def detect_v2(self, captcha, model):
        """
            Функция распознавания капчи V2.

            Args:
                captcha: Капча в виде массива NumPy.
                model: Модель YOLOv8, загруженная в память.

            Returns:
                Последовательность координат объектов, если все объекты были распознаны, иначе None.
        """
        prediction = model.predict(captcha)
        return prediction

    def predict_one_sample(self, model, inputs):
        """
            Проводит одно предсказание с использованием модели.

            Args:
                model: Модель, используемая для предсказания.
                inputs: Входные данные для предсказания в формате PyTorch Tensor.

            Returns:
                Вероятности предсказанных классов.
        """
        with torch.no_grad():
            inputs = inputs
            model.eval()
            logit = model(inputs).cpu()
            probs = torch.nn.functional.softmax(logit, dim=-1).numpy()
        return probs

    def classify_image(self, image_input):
        """
            Классифицирует изображение с помощью нейронной сети AlexNet.

            Args:
                image_input: Изображение для классификации в формате NumPy.

            Returns:
                Признак объекта, изображенного на капче.
        """
        # Список всех возможных признаков
        your_classes = [
            "arrow",
            "book",
            "bucket",
            "clock",
            "cloud",
            "compass",
            "electro",
            "eye",
            "face",
            "factory",
            "fire",
            "flag",
            "hand",
            "heart",
            "house",
            "key",
            "keyboard",
            "light",
            "lightning",
            "lock",
            "magnifier",
            "mail",
            "microphone",
            "monitor",
            "paper",
            "paperclip",
            "pen",
            "person",
            "photo",
            "pill",
            "scissors",
            "shop_cart",
            "sound",
            "star",
            "store_cart",
            "t-shirt",
            "ticket",
            "traffic_light",
            "umbrella",
            "water",
            "wrench",
        ]
        label_encoder = LabelEncoder()
        label_encoder.fit(your_classes)

        alexnet = AlexNet()
        alexnet.load_state_dict(torch.load("captcha_resolver/AI_weights/smartsolver_weights_1_6.pth", 
                                           map_location="cpu",))
        alexnet.eval()

        model = alexnet

        preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((227, 227)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        input_tensor = preprocess(image_input).unsqueeze(0)
        probs = self.predict_one_sample(model, input_tensor)
        predicted_class_idx = np.argmax(probs, axis=1)[0]
        return your_classes[predicted_class_idx]

    def put_object_to_s3(self, new_object, content):
        """
            Функция для загрузки объекта в Yandex Cloud Object Storage.

            Args:
                new_object: Имя объекта для загрузки.
                content: Содержимое объекта для загрузки.

            Returns:
                None
        """
        session = boto3.session.Session()
        s3 = session.client(
            service_name="s3",
            endpoint_url="https://storage.yandexcloud.net",
            aws_access_key_id=Config.aws_access_key_id,
            aws_secret_access_key=Config.aws_secret_access_key,)
        s3.put_object(Bucket="capchas-bucket", Key=new_object, Body=content, StorageClass="COLD")

    def get_batch(self):
        """
            Функция для скачивания и архивирования капчей из хранилища Yandex Cloud Object Storage.

            Returns:
                zip-архив с captchas.zip.
        """
        os.mkdir("download_captchas")
        session = boto3.session.Session()
        s3 = session.client(
            service_name="s3",
            endpoint_url="https://storage.yandexcloud.net",
            aws_access_key_id=Config.aws_access_key_id,
            aws_secret_access_key=Config.aws_secret_access_key)
        for key in s3.list_objects(Bucket="capchas-bucket")["Contents"]:
            print(key["Key"])
            get_object_response = s3.get_object(Bucket="capchas-bucket", Key=key["Key"])

            with open("download_captchas/" + key["Key"].split("/")[-1][:-4] + ".png", "wb") as fh:
                fh.write(base64.decodebytes(get_object_response["Body"].read()))
        shutil.make_archive("captchas", "zip", "download_captchas")
        shutil.rmtree("download_captchas")

    def delete_captchas(self):
        """
            Функция для удаления капчей из хранилища Yandex Cloud Object Storage.

            Returns:
                Статус удаления.
        """
        session = boto3.session.Session()
        s3 = session.client(
            service_name="s3",
            endpoint_url="https://storage.yandexcloud.net",
            aws_access_key_id=Config.aws_access_key_id,
            aws_secret_access_key=Config.aws_secret_access_key,
        )
        objects = s3.list_objects(Bucket="capchas-bucket", Prefix="captchas/")
        for object in objects["Contents"]:
            s3.delete_object(Bucket="capchas-bucket", Key=object["Key"])
        return {"status": "deleted"}

    def get_captcha_solve_sequence_segmentation_sobel(self, captcha, icons):
        """
            Функция для распознавания капчи с помощью сегментации и детектора Собеля.

            Args:
                captcha: Изображение капчи в формате NumPy.
                icons: Список иконок для поиска.

            Returns:
                Последовательность координат объектов, если все объекты были распознаны, иначе None.
        """
        copy = captcha.copy()
        sequence = []
        index = 1
        model = segmentation_model
        prediction = self.detect_v2(captcha, model)
        for icon in icons:
            name = self.classify_image(icon)
            x, y = self.get_boxes_detection(name, prediction, model)

            if x is not None and x != "not":
                cv2.circle(copy, (int(x), int(y)), 2, (0, 0, 255), 4)
                cv2.putText(
                    copy,
                    str(index),
                    (int(x) + 5, int(y) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,)

            sequence.append({"x": x, "y": y})
            index += 1
        return sequence

    def get_captcha_solve_sequence_hybrid_merge_business(self, request: RequestBusiness):
        """
            Функция для распознавания капчи с использованием гибридного метода и объединения результатов.

            Args:
                request: Объект запроса, содержащий изображение капчи и набор иконок.

            Returns:
                Последовательность координат объектов, если все объекты были распознаны, иначе None.

            Description:

            Функция `get_captcha_solve_sequence_hybrid_merge_business()` распознает капчу, используя гибридный метод,
            сочетающий сегментацию и детектор Собеля. Сначала функция распознает капчу с помощью сегментации и детектора Собеля,
            затем, если некоторые объекты не были обнаружены, она использует детектор Собеля для окончательного поиска и распознавания оставшихся объектов.

            Restriction:

            Функция требует наличия загруженных моделей `detection_model` и `segmentation_model`.

            Notes:

            Функция использует библиотеки `numpy` и `opencv-python` для обработки изображений и детектора Собеля.
        """
        captcha = b64_decode(request.screenshot_captcha)
        icons = preprocess_captcha_sobel(icons=b64_decode(request.screenshot_icons))
        sequence = []
        model = detection_model
        filtered_captcha = self.sobel_filter(request.filter, captcha)
        prediction = self.detect_v2(filtered_captcha, model)
        with ThreadPoolExecutor() as executor:
            # Запускаем все предсказания в отдельных потоках и собираем результаты
            result_xs_ys = list(executor.map(lambda icon: self.get_boxes_detection(self.classify_image(icon), prediction, model), icons))
            sequence = [{"x": i[0], "y": i[1]} for i in result_xs_ys]
        if any(i.get('x', False) is None for i in sequence):
            segment = self.get_captcha_solve_sequence_segmentation_sobel(captcha, icons)
            sequence, error = self.merge(sequence, segment)
        return sequence

    async def get_captcha_solve_sequence_hybrid_merge_business_async(self, request: RequestBusiness):
        """
            Асинхронное получение последовательности решения капчи методом гибридного слияния.

            Args:
                request: Запрос на решение капчи.

            Returns:
                Последовательность действий для решения капчи.
        """
        loop = asyncio.get_event_loop()
        # run_in_executor позволяет выполнить синхронную функцию асинхронно, возвращая Future
        # None в качестве первого аргумента означает использование стандартного пула исполнителей (executor)
        final_sequence = await loop.run_in_executor(None, self.get_captcha_solve_sequence_hybrid_merge_business, request)
        return final_sequence

    def merge(self, sequence: [dict], segment: [dict]):
        """
            Функция для объединения результатов распознавания капчи двумя разными методами.

            Args:
                sequence: Исходная последовательность координат объектов, полученная методом детектора Собеля.
                segment: Дополнительная последовательность координат объектов, полученная методом сегментации.

            Returns:
                Объединённая последовательность координат объектов, если все объекты были распознаны, иначе None.
        """
        final_sequence = []
        error = False
        for i in range(len(sequence)):
            if segment[i].get("x") is None and sequence[i].get("x") is not None:
                final_sequence.append(sequence[i])
            else:
                final_sequence.append(segment[i])
        for i in range(len(final_sequence)):
            if final_sequence[i].get("x") is None:
                error = True
        return final_sequence, error
