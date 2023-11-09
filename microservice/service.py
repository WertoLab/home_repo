import base64
import os
import shutil
import uuid
import boto3
import numpy

from microservice.preprocess import preprocess_captcha_sobel
from microservice.AI_models.ClassificationModel import AlexNet
import torch
import numpy as np
import cv2
import pickle
from numpy import sqrt
from numpy import sum
from torchvision import transforms
import microservice.controller as controller
from sklearn.preprocessing import LabelEncoder
from microservice.data.filters import RequestBusiness
import Config
import onnx
import onnxruntime as ort
from microservice.yolov8 import YOLOv8


def readb64(encoded_data):
    nparr = np.frombuffer(encoded_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return img


def b64_decode(im_b64: str):
    img_bytes = base64.b64decode(im_b64.encode("utf-8"))
    img = readb64(img_bytes)
    img_arr = np.asarray(img)
    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    return img_bgr


class Service:
    def get_onnx_inference(self, data):
        onnx_model = onnx.load("microservice/AI_weights/best_v3.onnx")
        onnx.checker.check_model(onnx_model)
        ort_sess = ort.InferenceSession("microservice/AI_weights/best_v3.onnx")
        captcha = b64_decode(data.screenshot_captcha)
        captcha = cv2.resize(captcha, (480, 480))
        captcha = self.sobel_filter(70, captcha)
        # icons = preprocess_captcha_sobel(icons=b64_decode(data.screenshot_icons))
        # outputs = ort_sess.run(None, {'images': captcha.reshape(1, 3, 480, 480).astype(numpy.float32)},)
        # print(np.array(outputs).shape)

        model_path = "microservice/AI_weights/captcha_segmentation.onnx"
        yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.8)

        # Read image

        # Detect Objects
        boxes, scores, class_ids = yolov8_detector(captcha)
        # print(boxes[0][0])
        for box in boxes:
            cv2.rectangle(
                captcha,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (255, 0, 0),
                2,
            )
        # print(boxes)
        # print(class_ids)
        cv2.imwrite("answer.png", captcha)
        return {"status": "ok"}

    def get_boxes(self, result):
        boxes = []
        all_params = result[0].boxes
        for i in range(len(result[0].boxes.conf.cpu())):
            if np.array(all_params.conf.cpu()[i]) > 0.05:
                x_up, y_up = (
                    all_params.xyxy.cpu()[i][0].numpy(),
                    all_params.xyxy.cpu()[i][1].numpy(),
                )
                x_bottom, y_bottom = (
                    all_params.xyxy.cpu()[i][2].numpy(),
                    all_params.xyxy.cpu()[i][3].numpy(),
                )
                boxes.append([x_up, y_up, x_bottom, y_bottom])

        return boxes

    def sobel_filter(self, threshold, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        G_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        G_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        rows, columns = img.shape
        mag = np.zeros(img.shape, dtype=np.float32)

        for i in range(0, rows - 2):
            for j in range(0, columns - 2):
                v = sum(sum(G_x * img[i : i + 3, j : j + 3]))
                h = sum(sum(G_y * img[i : i + 3, j : j + 3]))
                mag[i + 1, j + 1] = np.sqrt((v**2) + (h**2))
                if mag[i + 1, j + 1] < threshold:
                    mag[i + 1, j + 1] = 0

        processed_image = mag.astype(np.uint8)
        return cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)

    def get_boxes_detection(self, name, prediction, model):
        index = 0
        for box in self.get_boxes(prediction):
            if model.model.names[int(prediction[0].boxes.cls.cpu()[index])] == name:
                # int(box[0]), int(box[1]) , int(box[2]), int(box[3])
                print(box)
                return (int(box[2]) + int(box[0])) / 2, (int(box[3]) + int(box[1])) / 2
            index += 1
        return None, None

    def detect_v2(self, captcha, model):
        prediction = model.predict(captcha)

        return prediction

    def predict_one_sample(self, model, inputs):
        with torch.no_grad():
            inputs = inputs
            model.eval()
            logit = model(inputs).cpu()
            probs = torch.nn.functional.softmax(logit, dim=-1).numpy()
        return probs

    def classify_image(self, image_input):
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
        with open("microservice/label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)

        alexnet = AlexNet()
        alexnet.load_state_dict(
            torch.load(
                "microservice/AI_weights/smartsolver_weights_1_6.pth",
                map_location="cpu",
            )
        )
        alexnet.eval()

        label_encoder = pickle.load(open("microservice/label_encoder.pkl", "rb"))
        model = alexnet

        preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((227, 227)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        input_tensor = preprocess(image_input).unsqueeze(0)
        probs = self.predict_one_sample(model, input_tensor)
        predicted_class_idx = np.argmax(probs, axis=1)[0]

        if label_encoder:
            predicted_class = label_encoder.classes_[predicted_class_idx]
        else:
            predicted_class = str(predicted_class_idx)

        return predicted_class

    def put_object_to_s3(self, new_object, content):
        session = boto3.session.Session()
        s3 = session.client(
            service_name="s3",
            endpoint_url="https://storage.yandexcloud.net",
            aws_access_key_id=Config.aws_access_key_id,
            aws_secret_access_key=Config.aws_secret_access_key,
        )

        s3.put_object(
            Bucket="capchas-bucket", Key=new_object, Body=content, StorageClass="COLD"
        )

    def get_batch(self):
        os.mkdir("download_captchas")
        session = boto3.session.Session()
        s3 = session.client(
            service_name="s3",
            endpoint_url="https://storage.yandexcloud.net",
            aws_access_key_id=Config.aws_access_key_id,
            aws_secret_access_key=Config.aws_secret_access_key,
        )

        for key in s3.list_objects(Bucket="capchas-bucket")["Contents"]:
            print(key["Key"])
            get_object_response = s3.get_object(Bucket="capchas-bucket", Key=key["Key"])

            with open(
                "download_captchas/" + key["Key"].split("/")[-1][:-4] + ".png", "wb"
            ) as fh:
                fh.write(base64.decodebytes(get_object_response["Body"].read()))
        shutil.make_archive("captchas", "zip", "download_captchas")
        shutil.rmtree("download_captchas")

    def delete_captchas(self):
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

    def get_captcha_solve_sequence_segmentation_sobel(
        self, request: RequestBusiness, captcha, icons
    ):
        copy = captcha.copy()
        sequence = []
        index = 1
        detected_objects = 0
        captcha_id = str(uuid.uuid4())
        model = controller.segmentation_model
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
                    2,
                )

            sequence.append({"x": x, "y": y})
            index += 1
        """
        if detected_objects != index - 1:
            os.mkdir("captchas")
            print("saved")
            cv2.imwrite("captchas/" + captcha_id + ".png", b64_decode(request.screenshot_captcha))
            with open(str("captchas/" + captcha_id + ".png"), 'rb') as file:
                b64_string_captcha = base64.b64encode(file.read()).decode('UTF-8')
            self.put_object_to_s3("captchas/" + captcha_id + ".txt", b64_string_captcha)
            shutil.rmtree("captchas")
        """
        # b64_string_discolored = base64.b64encode(self.sobel_filter(70, captcha)).decode('UTF-8')
        # b64_string_answer = base64.b64encode(copy).decode('UTF-8')

        return sequence

    """
    def get_captcha_solve_sequence_hybrid(self, request: RequestSobel):
        captcha = b64_decode(request.screenshot_captcha)
        icons = preprocess_captcha_sobel(icons=b64_decode(request.screenshot_icons))
        copy = captcha.copy()
        sequence = []
        index = 1

        for icon in icons:
            name = self.classify_image(icon)
            x, y = self.detect_v2(name, captcha, request.filter.value, "best_v2.pt")
            if x is not None and x != "not":
                cv2.circle(copy, (int(x), int(y)), 2, (0, 0, 255), 4)
                cv2.putText(copy, str(index), (int(x) + 5, int(y) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            sequence.append({"order": index, "center_coordinates": {"x": x, "y": y}})
            index += 1

        b64_string_discolored = base64.b64encode(self.sobel_filter(70, captcha)).decode('UTF-8')
        b64_string_answer = base64.b64encode(copy).decode('UTF-8')
        cv2.imwrite("answer.png", copy)

        return sequence, b64_string_discolored, request.screenshot_captcha, request.screenshot_icons, b64_string_answer

    def get_captcha_solve_sequence_hybrid_merge(self, request: RequestSobel):
        captcha = b64_decode(request.screenshot_captcha)
        discolored_captcha, icons = preprocess_captcha_sobel(icons=b64_decode(request.screenshot_icons))
        copy = captcha.copy()
        sequence = []
        index = 1

        for icon in icons:
            name = self.classify_image(icon)
            x, y = self.detect_v2(name, captcha, request.filter.value, "best_v3.pt")
            sequence.append({"x": x, "y": y})
            index += 1

        final_sequence = []
        segment = self.get_captcha_solve_sequence_segmentation_sobel(request)[0]

        for i in range(5):
            if segment[i].get("x") is None and sequence[i].get("x") is not None:
                final_sequence.append(sequence[i])
            elif segment[i].get("x") is not None and sequence[i].get("x") is None:
                final_sequence.append(segment[i])
            else:
                final_sequence.append(segment[i])

        for i in range(5):
            if final_sequence[i].get("x") is not None:
                cv2.circle(copy, (int(final_sequence[i].get("x")), int(final_sequence[i].get("y"))), 2, (0, 0, 255), 4)
                cv2.putText(copy, str(i+1), (int(final_sequence[i].get("x")) + 5, int(final_sequence[i].get("y")) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        b64_string_discolored = base64.b64encode(self.sobel_filter(70, captcha)).decode('UTF-8')
        b64_string_answer = base64.b64encode(copy).decode('UTF-8')
        cv2.imwrite("answer.png", copy)

        return final_sequence, b64_string_discolored, request.screenshot_captcha, request.screenshot_icons, b64_string_answer
    """

    def get_captcha_solve_sequence_hybrid_merge_business(
        self, request: RequestBusiness
    ):
        captcha = b64_decode(request.screenshot_captcha)
        icons = preprocess_captcha_sobel(icons=b64_decode(request.screenshot_icons))
        copy = captcha.copy()
        sequence = []
        index = 1
        model = controller.detection_model
        filtered_captcha = self.sobel_filter(request.filter, captcha)
        prediction = self.detect_v2(filtered_captcha, model)
        for icon in icons:
            name = self.classify_image(icon)
            x, y = self.get_boxes_detection(name, prediction, model)
            sequence.append({"x": x, "y": y})
            index += 1

        final_sequence = []
        error = False
        segment = self.get_captcha_solve_sequence_segmentation_sobel(
            request, captcha, icons
        )

        for i in range(len(sequence)):
            if segment[i].get("x") is None and sequence[i].get("x") is not None:
                final_sequence.append(sequence[i])
            else:
                final_sequence.append(segment[i])
        for i in range(len(sequence)):
            if final_sequence[i].get("x") is None:
                error = True
            if final_sequence[i].get("x") is not None:
                cv2.circle(
                    copy,
                    (int(final_sequence[i].get("x")), int(final_sequence[i].get("y"))),
                    2,
                    (0, 0, 255),
                    4,
                )
                cv2.putText(
                    copy,
                    str(i + 1),
                    (
                        int(final_sequence[i].get("x")) + 5,
                        int(final_sequence[i].get("y")) + 4,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        # b64_string_discolored = base64.b64encode(filtered_captcha).decode('UTF-8')
        # b64_string_answer = base64.b64encode(copy).decode('UTF-8')
        # cv2.imwrite("answer.png", copy)

        return final_sequence, error
