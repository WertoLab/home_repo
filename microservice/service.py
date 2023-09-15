import base64
import os
import shutil
import uuid

import boto3

from microservice.preprocess import preprocess_captcha_v2
from microservice.preprocess import preprocess_captcha_v2_business
from microservice.AI_models.ClassificationModel import AlexNet
import torch
import numpy as np
import cv2
import pickle
from torchvision import transforms
from ultralytics import YOLO
from sklearn.preprocessing import LabelEncoder
from microservice.data.filters import RequestSobel, RequestDiscolor, RequestImagesOnly


def readb64(encoded_data):
    nparr = np.frombuffer(encoded_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return img


class Service:
    def __init__(self):
        pass

    def get_boxes(self, result):
        boxes = []
        all_params = result[0].boxes
        for i in range(len(result[0].boxes.conf.cpu())):
            if (np.array(all_params.conf.cpu()[i]) > 0.05):
                x_up, y_up = all_params.xyxy.cpu()[i][0].numpy(), all_params.xyxy.cpu()[i][1].numpy()
                x_bottom, y_bottom = all_params.xyxy.cpu()[i][2].numpy(), all_params.xyxy.cpu()[i][3].numpy()
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
                v = np.sum(np.sum(G_x * img[i:i + 3, j:j + 3]))  # vertical
                h = np.sum(np.sum(G_y * img[i:i + 3, j:j + 3]))  # horizon
                mag[i + 1, j + 1] = np.sqrt((v ** 2) + (h ** 2))

        for p in range(0, rows):
            for q in range(0, columns):
                if mag[p, q] < threshold:
                    mag[p, q] = 0

        processed_image = mag.astype(np.uint8)
        return cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)

    def detect_v2(self, name, captcha, threshold, model_name):
        model = YOLO("microservice/AI_weights/" + str(model_name))
        prediction = model.predict(self.sobel_filter(threshold, captcha))
        # cv2.imwrite("sobel.png", self.sobel_filter(threshold, captcha))
        index = 0
        for box in self.get_boxes(prediction):
            if model.model.names[int(prediction[0].boxes.cls.cpu()[index])] == name:
                # int(box[0]), int(box[1]) , int(box[2]), int(box[3])
                print(box)
                return (int(box[2]) + int(box[0])) / 2, (int(box[3]) + int(box[1])) / 2
            index += 1
        return None, None

    def detect_v1(self, name, captcha):
        model = YOLO("microservice/AI_weights/best.pt")
        prediction = model.predict(captcha)
        index = 0
        for box in self.get_boxes(prediction):
            if model.model.names[int(prediction[0].boxes.cls.cpu()[index])] == name:
                # int(box[0]), int(box[1]) , int(box[2]), int(box[3])
                print(box)
                return (int(box[2]) + int(box[0])) / 2, (int(box[3]) + int(box[1])) / 2
            index += 1
        return None, None

    def predict_one_sample(self, model, inputs):

        with torch.no_grad():
            inputs = inputs
            model.eval()
            logit = model(inputs).cpu()
            probs = torch.nn.functional.softmax(logit, dim=-1).numpy()
        return probs

    def classify_image(self, image_input):
        your_classes = ["arrow", "book", "bucket", "clock", "cloud", "compass", "electro", "eye", "face",
                        "factory", "fire", "flag", "hand", "heart", "house", "key", "keyboard", "light",
                        "lightning", "lock", "magnifier", "mail", "microphone", "monitor", "paper",
                        "paperclip", "pen", "person", "photo", "pill", "scissors", "shop_cart", "sound",
                        "star", "store_cart", "t-shirt", "ticket", "traffic_light", "umbrella", "water", "wrench"]
        label_encoder = LabelEncoder()
        label_encoder.fit(your_classes)
        with open("microservice/label_encoder.pkl", "wb") as f:
            pickle.dump(label_encoder, f)

        # Загрузите вашу обученную модель
        alexnet = AlexNet()
        alexnet.load_state_dict(torch.load("microservice/AI_weights/smartsolver_weights_1_6.pth", map_location='cpu'))
        alexnet.eval()

        # Загрузите label_encoder, если используете его
        label_encoder = pickle.load(open("microservice/label_encoder.pkl", 'rb'))
        model = alexnet

        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(image_input).unsqueeze(0)
        probs = self.predict_one_sample(model, input_tensor)
        predicted_class_idx = np.argmax(probs, axis=1)[0]
        # print(predicted_class_idx)
        if label_encoder:
            predicted_class = label_encoder.classes_[predicted_class_idx]
        else:
            predicted_class = str(predicted_class_idx)

        return predicted_class

    def b64_decode(self, im_b64: str):
        img_bytes = base64.b64decode(im_b64.encode('utf-8'))
        img = readb64(img_bytes)
        img_arr = np.asarray(img)
        img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        return img_bgr

    def put_object_to_s3(self, new_object, content):
        session = boto3.session.Session()
        s3 = session.client(
            service_name='s3',
            endpoint_url='https://storage.yandexcloud.net',
            aws_access_key_id='YCAJEPC30tPiNB3wctwCuqNhZ',
            aws_secret_access_key='YCOCKWm4HIFDBMV4-jNTtTe20QQHAx42NPJkdkI8'
        )
        print(content)
        s3.put_object(Bucket='capchas-bucket', Key=new_object, Body=content,
                      StorageClass='COLD')

    def get_batch(self):
        os.mkdir("download_captchas")

        session = boto3.session.Session()
        s3 = session.client(
            service_name='s3',
            endpoint_url='https://storage.yandexcloud.net',
            aws_access_key_id='YCAJEPC30tPiNB3wctwCuqNhZ',
            aws_secret_access_key='YCOCKWm4HIFDBMV4-jNTtTe20QQHAx42NPJkdkI8'
        )

        for key in s3.list_objects(Bucket='capchas-bucket')['Contents']:
            print(key['Key'])
            get_object_response = s3.get_object(Bucket='capchas-bucket', Key=key['Key'])

            with open("download_captchas/" + key['Key'].split("/")[-1][:-4] + ".png", "wb") as fh:
                fh.write(base64.decodebytes(get_object_response['Body'].read()))
        shutil.make_archive('captchas', 'zip', 'download_captchas')
        shutil.rmtree("download_captchas")

    def delete_captchas(self):
        session = boto3.session.Session()
        s3 = session.client(
            service_name='s3',
            endpoint_url='https://storage.yandexcloud.net',
            aws_access_key_id='YCAJEPC30tPiNB3wctwCuqNhZ',
            aws_secret_access_key='YCOCKWm4HIFDBMV4-jNTtTe20QQHAx42NPJkdkI8'
        )
        objects = s3.list_objects(Bucket='capchas-bucket', Prefix='captchas/')
        for object in objects['Contents']:
            s3.delete_object(Bucket='capchas-bucket', Key=object['Key'])
        return {"stutus": "deleted"}

    def get_captcha_solve_sequence_old(self, request: RequestImagesOnly):
        captcha = self.b64_decode(request.screenshot_captcha)
        discolored_captcha, icons = preprocess_captcha_v2(img=captcha.copy(),
                                                          icons=self.b64_decode(request.screenshot_icons))
        copy = captcha.copy()
        sequence = []
        index = 1
        for icon in icons:
            name = self.classify_image(icon)
            # print(name)
            # x, y = self.detect(name, discolored_captcha)
            x, y = self.detect_v1(name, discolored_captcha)

            if (x != None and x != "not"):
                cv2.circle(copy, (int(x), int(y)), 2, (0, 0, 255), 4)
                cv2.putText(copy, str(index), (int(x) + 5, int(y) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            sequence.append({"order": index, "center_coordinates": {"x": x, "y": y}})
            index += 1
        '''
        if (json.get("type") == "algorythm_1"):
            # x, y = self.detect(name, discolored_captcha)
            cv2.imwrite("/Users/andrey/Desktop/soutions/old_discolor/answer1.png", copy)
        else:
            cv2.imwrite("/Users/andrey/Desktop/soutions/sobel/answer.png", copy)
        '''
        # print(sequence)
        # print(self.detect("face", cv2.imread("preprocesses_captcha0.png")))
        return sequence

    def get_captcha_solve_sequence_sobel(self, request: RequestImagesOnly):
        captcha = self.b64_decode(request.screenshot_captcha)
        discolored_captcha, icons = preprocess_captcha_v2(img=captcha.copy(),
                                                          icons=self.b64_decode(request.screenshot_icons))
        copy = captcha.copy()
        sequence = []
        index = 1
        for icon in icons:
            name = self.classify_image(icon)
            x, y = self.detect_v2(name, captcha, 70, "best_custom.pt")

            if (not x is None) and x != "not":
                cv2.circle(copy, (int(x), int(y)), 2, (0, 0, 255), 4)
                cv2.putText(copy, str(index), (int(x) + 5, int(y) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            sequence.append({"order": index, "center_coordinates": {"x": x, "y": y}})
            index += 1
        return sequence

    def get_captcha_solve_sequence_old_business(self, request: RequestDiscolor):
        captcha = self.b64_decode(request.screenshot_captcha)
        discolored_captcha, icons = preprocess_captcha_v2_business(img=captcha,
                                                                   icons=self.b64_decode(request.screenshot_icons),
                                                                   filter=request.filter)
        copy = captcha.copy()
        sequence = []
        index = 1
        detected_objects = 0
        captcha_id = str(uuid.uuid4())
        for icon in icons:
            name = self.classify_image(icon)
            x, y = self.detect_v1(name, discolored_captcha)
            if (x != None and x != "not"):
                cv2.circle(copy, (int(x), int(y)), 2, (0, 0, 255), 4)
                cv2.putText(copy, str(index), (int(x) + 5, int(y) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            sequence.append({"order": index, "center_coordinates": {"x": x, "y": y}})
            index += 1
        if (detected_objects != index - 1):
            os.mkdir("captchas")
            print("saved")
            cv2.imwrite("captchas/" + captcha_id + ".png", self.b64_decode(request.screenshot_captcha))
            with open(str("captchas/" + captcha_id + ".png"), 'rb') as file:
                b64_string_captcha = base64.b64encode(file.read()).decode('UTF-8')
            self.put_object_to_s3("captchas/" + captcha_id + ".txt", b64_string_captcha)
            shutil.rmtree("captchas")
        b64_string_discolored = base64.b64encode(discolored_captcha).decode('UTF-8')
        b64_string_answer = base64.b64encode(copy).decode('UTF-8')
        return sequence, b64_string_discolored, request.screenshot_captcha, request.screenshot_icons, b64_string_answer

    def get_captcha_solve_sequence_sobel_business(self, request: RequestSobel):
        captcha = self.b64_decode(request.screenshot_captcha)
        discolored_captcha, icons = preprocess_captcha_v2(img=captcha,
                                                          icons=self.b64_decode(request.screenshot_icons))
        copy = captcha.copy()
        sequence = []
        index = 1
        detected_objects = 0
        captcha_id = str(uuid.uuid4())
        for icon in icons:
            name = self.classify_image(icon)
            print(name)
            x, y = self.detect_v2(name, captcha, request.filter.value, "best_custom.pt")

            if (x != None and x != "not"):
                cv2.circle(copy, (int(x), int(y)), 2, (0, 0, 255), 4)
                cv2.putText(copy, str(index), (int(x) + 5, int(y) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            sequence.append({"order": index, "center_coordinates": {"x": x, "y": y}})
            index += 1
        '''
        if (json.get("type") == "algorythm_1"):
            # x, y = self.detect(name, discolored_captcha)
            cv2.imwrite("/Users/andrey/Desktop/soutions/old_discolor/answer1.png", copy)
        else:
            cv2.imwrite("/Users/andrey/Desktop/soutions/sobel/answer.png", copy)
        '''
        if (detected_objects != index - 1):
            os.mkdir("captchas")
            print("saved")
            cv2.imwrite("captchas/" + captcha_id + ".png", self.b64_decode(request.screenshot_captcha))
            with open(str("captchas/" + captcha_id + ".png"), 'rb') as file:
                b64_string_captcha = base64.b64encode(file.read()).decode('UTF-8')
            self.put_object_to_s3("captchas/" + captcha_id + ".txt", b64_string_captcha)
            shutil.rmtree("captchas")
        # print(sequence)
        b64_string_discolored = base64.b64encode(self.sobel_filter(70, captcha)).decode('UTF-8')
        b64_string_answer = base64.b64encode(copy).decode('UTF-8')
        # print(self.detect("face", cv2.imread("preprocesses_captcha0.png")))
        return sequence, b64_string_discolored, request.screenshot_captcha, request.screenshot_icons, b64_string_answer
