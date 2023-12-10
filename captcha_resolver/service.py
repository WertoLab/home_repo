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


from ultralytics import YOLO

from captcha_resolver import init_models
from captcha_resolver.yolov8 import YOLOv8

segmentation_model: YOLO
detection_model: YOLO
segmentation_onnx_model: YOLOv8
detection_onnx_model: YOLOv8
segmentation_model, detection_model, segmentation_onnx_model, detection_onnx_model, alexnet = init_models()

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
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


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
    return cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)


class Service:
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
            if your_classes[class_ids[i]] == name:
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
        all_params = result[0].boxes
        # Перемещаем данные в CPU и преобразуем в NumPy один раз, чтобы избежать многократных вызовов
        confs = all_params.conf.cpu().numpy()
        boxes = all_params.xyxy.cpu().numpy()
        # Используем маску для фильтрации рамок с уверенностью выше порога
        mask = confs > 0.05
        filtered_boxes = boxes[mask]
        # Преобразуем полученные рамки в нужный формат
        return filtered_boxes[:, :4].tolist()  # [:4] включает x_up, y_up, x_bottom, y_bottom

    def sobel_filter(self, threshold, img):
        """
            Применяет фильтр Собеля к изображению, чтобы выделить края.

            Args:
                threshold: Значение порога для определения значащих краев.
                img: Исходное изображение в формате NumPy.

            Returns:
                Обработанное изображение с выделенными краями.
        """
        def efficient_sobel(img, threshold):
            # Конвертация изображения в оттенки серого
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Применение фильтра Собеля для вычисления градиента по оси X и Y
            sobelx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
            # Вычисление магнитуды градиента
            mag = np.sqrt(sobelx ** 2 + sobely ** 2)
            # Аналог паддинга нулями
            mag[0, :] = mag[-1, :] = mag[:, 0] = mag[:, -1] = 0
            # Применение порога
            _, mag_thresh = cv2.threshold(mag, threshold, 255, cv2.THRESH_TOZERO)
            # Преобразование магнитуды градиента в 8-битное изображение
            # Используем нормализацию для преобразования в диапазон 0-255
            processed_image = cv2.cvtColor(mag_thresh.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            return processed_image

        efficient = efficient_sobel(img, threshold)
        # Временно закомментируем эти строки, так как перешли на более эффективный фильтр Собеля
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # G_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        # G_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        # rows, columns = img.shape
        # mag = np.zeros(img.shape, dtype=np.float32)
        # for i in range(0, rows - 2):
        #     for j in range(0, columns - 2):
        #         v = sum(sum(G_x * img[i: i + 3, j: j + 3]))
        #         h = sum(sum(G_y * img[i: i + 3, j: j + 3]))
        #         mag[i + 1, j + 1] = np.sqrt((v ** 2) + (h ** 2))
        #         if mag[i + 1, j + 1] < threshold:
        #             mag[i + 1, j + 1] = 0
        # processed_image = mag.astype(np.uint8)
        # res = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
        return efficient

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
        return model.predict(captcha)

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
        input_tensor = preprocess(image_input).unsqueeze(0)
        res = your_classes[np.argmax(self.predict_one_sample(alexnet, input_tensor), axis=1)[0]]
        return res

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
        model = segmentation_model
        prediction = self.detect_v2(captcha, model)
        for index, icon in enumerate(icons):
            index += 1
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
        import time
        a = [time.time()]
        b = []
        captcha = b64_decode(request.screenshot_captcha)
        a.append(time.time())
        b.append("b64")
        icons = preprocess_captcha_sobel(icons=b64_decode(request.screenshot_icons))
        a.append(time.time())
        b.append("preprocess")
        filtered_captcha = self.sobel_filter(request.filter, captcha)
        a.append(time.time())
        b.append("sobel")
        prediction = self.detect_v2(filtered_captcha, detection_model)
        a.append(time.time())
        b.append("detection")
        result_xs_ys = []
        for num, icon in enumerate(icons):
            to_box = self.classify_image(icon)
            a.append(time.time())
            b.append("classification " + str(num+1))
            result_xs_ys.append(self.get_boxes_detection(to_box, prediction, detection_model))
            a.append(time.time())
            b.append("get_boxes_detection " + str(num+1))
        # result_xs_ys = [self.get_boxes_detection(self.classify_image(icon),
        #                                          prediction,
        #                                          detection_model) for icon in icons]
        sequence = [{"x": i[0], "y": i[1]} for i in result_xs_ys]
        if any(i.get('x', False) is None for i in sequence):
            segment = self.get_captcha_solve_sequence_segmentation_sobel(captcha, icons)
            sequence, error = self.merge(sequence, segment)
        for i in range(len(b)):
            print(f"{b[i]}: {a[i+1] - a[i]:.2f}")
        print(f"Total time: {a[-1] - a[0]:.2f}")
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
        with ThreadPoolExecutor() as pool:
            final_sequence = await loop.run_in_executor(pool,
                                                        self.get_captcha_solve_sequence_hybrid_merge_business,
                                                        request)
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
