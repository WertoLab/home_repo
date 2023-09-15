import ast
import base64
import json
import os.path
import sys
from pathlib import Path
from typing import Union
import cv2
import requests as r

ROOT_PATH = Path(__file__).resolve(strict=True).parent.parent.parent
YPATHCAPTCHA="/Users/andrey/Desktop/test_samples/captcha2/image_FON.png"
YPATHICONS="/Users/andrey/Desktop/test_samples/captcha2/image_ICONS.png"
with open(str(YPATHCAPTCHA), 'rb') as file:
    b64_string_captcha = base64.b64encode(file.read()).decode('UTF-8')

with open(str(YPATHICONS), 'rb') as file:
    b64_string_icons = base64.b64encode(file.read()).decode('UTF-8')

file_old_business = {"screenshot_captcha": b64_string_captcha, "screenshot_icons": b64_string_icons, "filter": {"step": 2, "tolerance": 6, "count_contour": 1400, "blur": True}}
file_sobel_business = {"screenshot_captcha": b64_string_captcha, "screenshot_icons": b64_string_icons, "filter": { "value":200}}
file_old_our = {"screenshot_captcha": b64_string_captcha, "screenshot_icons": b64_string_icons}
file_sobel_our = {"screenshot_captcha": b64_string_captcha, "screenshot_icons": b64_string_icons}

data = json.dumps(file_old_business)

#print(data)
headers = {"Content-type": "application/json", "Accept": "text/plain"}
response = r.get('http://localhost:8000/get_captcha_solve_sequence_old_business', data=data, headers=headers)
coord_str = response.content.decode('UTF-8')
result_path = '/Users/andrey/Desktop/ dataset/159.jpg'
copy = cv2.imread('/Users/andrey/Desktop/ dataset/69.jpg')
cv2.rectangle(copy, (int(0), int(217)), (int(440), int(300)), (255, 0, 0), 2)
#cv2.rectangle(copy, (int(8), int(441)), (int(338), int(514)), (255, 0, 0), 2)
cv2.imwrite(result_path, copy)
print(coord_str)



