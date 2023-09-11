import cv2

'''
import captcha_transformation_2.v1.split_captcha.example
from captcha_transformation_2.v1.discolor_captcha.discolor_captcha import discolor
from captcha_transformation_2.v1.discolor_captcha.example import save_result
from captcha_transformation_2.v1.split_captcha.split_captcha import split_captcha
from captcha_transformation_2.values import Color
'''
from imutils import paths
import captcha_transformation_2.v2.split_captcha.example
from captcha_transformation_2.v2.discolor_captcha.discolor_captcha import discolor
from captcha_transformation_2.v2.discolor_captcha.example import save_result
from captcha_transformation_2.values import Color


def init_dataset():
    "Или по другому как вам угодно"
    return captcha_transformation_2.v1.split_captcha.example.init_dataset()


def preprocess_captcha_v2_business(img, icons, json: dict):
    # images = init_dataset()
    # print("ok")
    return_img = discolor(image=img, step=json.get("discolor_filter").get("step"),
                          tol=json.get("discolor_filter").get("tolerance"),
                          count_contour=json.get("discolor_filter").get("count_contour"), new_tint_icon=Color.WHITE,
                          blur=True)
    # print(list(paths.list_images("/Users/andrey/Downloads/five_boys")))
    icons_d = captcha_transformation_2.v2.split_captcha.split_captcha.split_icons(icons)
    ans = captcha_transformation_2.v2.split_captcha.split_captcha.create_icons(image=icons, icons=icons_d)

    return return_img, ans


def preprocess_captcha_v2(img, icons):
    # images = init_dataset()
    # print("ok")
    return_img = discolor(image=img, step=2,
                          tol=6,
                          count_contour=1400, new_tint_icon=Color.WHITE,
                          blur=True)
    # print(list(paths.list_images("/Users/andrey/Downloads/five_boys")))
    icons_d = captcha_transformation_2.v2.split_captcha.split_captcha.split_icons(icons)
    ans = captcha_transformation_2.v2.split_captcha.split_captcha.create_icons(image=icons, icons=icons_d)

    return return_img, ans
