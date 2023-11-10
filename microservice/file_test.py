import base64
import json
from pathlib import Path
import cv2
import requests as r

ROOT_PATH = Path(__file__).resolve(strict=True).parent.parent.parent
YPATHCAPTCHA = "./images/msg803864447-15472.jpg"  # 9
YPATHICONS = "./images/msg803864447-15473.jpg"

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
# REQUEST_PATH = "http://95.31.6.30:8000/get_captchas"
# REQUEST_PATH1 = "http://localhost:8000/hello"
LOCAL_REQUEST_PATH = "http://127.0.0.1:8000/get_captchas"

# print(data)
headers = {"Content-type": "application/json", "Accept": "text/plain"}
response = r.post(LOCAL_REQUEST_PATH, data=data, headers=headers)
coord_str = response.content.decode("UTF-8")
# result_path = "/Users/andrey/Desktop/ dataset/159.jpg"
# copy = cv2.imread("/Users/andrey/Desktop/ dataset/69.jpg")
# cv2.rectangle(copy, (int(0), int(217)), (int(440), int(300)), (0, 0, 255), 2)
# # cv2.rectangle(copy, (int(8), int(441)), (int(338), int(514)), (255, 0, 0), 2)
# cv2.imwrite(result_path, copy)
# print(response.status_code)
print(coord_str)
