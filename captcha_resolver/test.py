import base64

import cv2
import numpy as np
import onnx
import onnxruntime as ort


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


onnx_model = onnx.load("AI_weights/best_v3.onnx")
onnx.checker.check_model(onnx_model)
ort_sess = ort.InferenceSession("AI_weights/best_v3.onnx")

YPATHCAPTCHA = "/Users/andrey/Downloads/sotka/image_001.png"  # 9

with open(str(YPATHCAPTCHA), "rb") as file:
    b64_string_captcha = base64.b64encode(file.read()).decode("UTF-8")
captcha = b64_decode(b64_string_captcha)
captcha = cv2.resize(captcha, (480, 480))
# captcha = self.sobel_filter(70, captcha)
# icons = preprocess_captcha_sobel(icons=b64_decode(data.screenshot_icons))
print(ort_sess.get_inputs()[0])
print(ort_sess.get_outputs()[0])
outputs = ort_sess.run(
    None, {"images": captcha.reshape(1, 3, 480, 480).astype(np.float32)}
)
