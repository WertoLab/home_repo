import base64
import concurrent.futures
import threading

import cv2
import numpy as np
import onnx
import onnxruntime as ort

'''
onnx_model = onnx.load("AI_weights/best_v3.onnx")
onnx.checker.check_model(onnx_model)
ort_sess = ort.InferenceSession("AI_weights/best_v3.onnx")
'''

'''
print(ort_sess.get_inputs()[0])
print(ort_sess.get_outputs()[0])
outputs = ort_sess.run(
    None, {"images": captcha.reshape(1, 3, 480, 480).astype(np.float32)}
)
'''

import base64
import json
from pathlib import Path
import requests as r

ROOT_PATH = Path(__file__).resolve(strict=True).parent.parent.parent
YPATHCAPTCHA = "/Users/andrey/Downloads/sotka/image_020.png"  # 9
YPATHICONS = "/Users/andrey/Downloads/sotka/5020.png"

with open(str(YPATHCAPTCHA), "rb") as file:
    b64_string_captcha = base64.b64encode(file.read()).decode("UTF-8")

with open(str(YPATHICONS), "rb") as file:
    b64_string_icons = base64.b64encode(file.read()).decode("UTF-8")

file_old_business = {
    "screenshot_captcha": b64_string_captcha,
    "screenshot_icons": b64_string_icons,
    "filter": {"step": 2, "tolerance": 6, "count_contour": 1400, "blur": True},
}
file_sobel_business = {
    "screenshot_captcha": b64_string_captcha,
    "screenshot_icons": b64_string_icons,
    "filter": {"value": 70},
}
file_old_our = {
    "screenshot_captcha": b64_string_captcha,
    "screenshot_icons": b64_string_icons,
}
file_sobel_our = {
    "screenshot_captcha": b64_string_captcha,
    "screenshot_icons": b64_string_icons,
}
file_business = {
    "method": "base64",
    "coordinatescaptcha": 1,
    "key": "YOUR_APIKEY",
    "body": b64_string_captcha,
    # "body": "1111111111111111111",
    "imginstructions": b64_string_icons,
    "textinstructions": "Кликните в таком порядке | Click in the following order",
    "json": 1,
    "sobel_filter": 70,
}
data = json.dumps(file_business)
REQUEST_PATH = "http://95.31.6.30:8000/get_captchas"
REQUEST_PATH1 = "http://localhost:8000/hello"
LOCAL_REQUEST_PATH = "http://localhost:8000/get_captchas"

# print(data)
headers = {"Content-type": "application/json", "Accept": "text/plain"}


def post():
    response = r.post(REQUEST_PATH, data=data, headers=headers)
    coord_str = response.content.decode("UTF-8")
    print(coord_str)

times = []

for i in range(1,1000):
    times.append(i)
while(True):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        times = []
        for i in range(1, 1000):
            times.append(i)
        results = [executor.submit(post()) for time in times]

        for f in concurrent.futures.as_completed(results):
            print(f.result())

x = threading.Thread(target=post(), args=())
x.start()
# result_path = "/Users/andrey/Desktop/ dataset/159.jpg"
# copy = cv2.imread("/Users/andrey/Desktop/ dataset/69.jpg")
# cv2.rectangle(copy, (int(0), int(217)), (int(440), int(300)), (0, 0, 255), 2)
# # cv2.rectangle(copy, (int(8), int(441)), (int(338), int(514)), (255, 0, 0), 2)
# cv2.imwrite(result_path, copy)
# print(response.status_code)
# print(coord_str)

'''
upstream loadbalancer {
    server app1:8000 weight=5;
    server app2:8000 weight=5;
}

server {
    location / {
        proxy_pass http://loadbalancer;
    }
}




 microservice3:
        build:
            context: .
        restart: always
        expose:
            - "8002"

    microservice4:
        build:
            context: .
        restart: always
        expose:
            - "8003"

    microservice5:
        build:
            context: .
        restart: always
        expose:
            - "8004"


    - microservice3
            - microservice4
            - microservice5





upstream loadbalancer3 {
    server microservice3:8002;
}

upstream loadbalancer4 {
    server microservice4:8003;
}

upstream loadbalancer5{
    server microservice5:8004;

}



server {
        listen 80;
        location / {
          proxy_pass    http://loadbalancer3;
        }
}

server {
        listen 80;
        location / {
          proxy_pass    http://loadbalancer4;
        }
}

server {
        listen 80;
        location / {
          proxy_pass    http://loadbalancer5;
        }
}


'''
