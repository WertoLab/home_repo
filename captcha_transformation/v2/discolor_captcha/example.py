from captcha_transformation.v2.discolor_captcha.discolor_captcha import discolor
from captcha_transformation.values import Color
import os
import cv2

YOUR_PATH_DATASET = "/dataset"
YOUR_PATH_SAVE_RESULT = "dataset/result"


def init_dataset():
    images = []
    for i in range(0,2):
        images.append( cv2.imread(f"{YOUR_PATH_DATASET}/captcha_{i}.png",cv2.IMREAD_UNCHANGED))
    return images


def save_result(res, step, tol, count_contour):
    for i in range(0, len(res)):
        image = res[i]
        index = str(i+1).zfill(4)
        os.makedirs(
            os.path.dirname(f"{YOUR_PATH_SAVE_RESULT}/Шаг_{step + 1}__Погрешность_{tol}_Контуров_{count_contour}/"),
            exist_ok=True)
        cv2.imwrite(f"{YOUR_PATH_SAVE_RESULT}/Шаг_{step + 1}__Погрешность_{tol}_Контуров_{count_contour}/captcha{index}.png",
                    image)


if __name__ == "__main__":
    dataset = init_dataset()
    res = []
    for img in dataset:
        if not (img is None):
            res.append(discolor(image=img, step=5, tol=3, count_contour=100, new_tint_icon=Color.WHITE, blur=True))
    for i in res:
        cv2.imshow("i", i)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    if len(res) > 0:
        save_result(res=res, step=1, tol=5, count_contour=100)
