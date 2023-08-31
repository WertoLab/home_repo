import cv2
import captcha_transformation.v1.split_captcha.example
from captcha_transformation.v1.discolor_captcha.discolor_captcha import discolor
from captcha_transformation.v1.discolor_captcha.example import save_result
from captcha_transformation.v1.split_captcha.split_captcha import split_captcha
from captcha_transformation.values import Color


def init_dataset():
    "Или по другому как вам угодно"
    return captcha_transformation.v1.split_captcha.example.init_dataset()


if __name__ == "__main__":
    images = init_dataset()
    datasets_split_captcha = []
    for i in range(1, len(images) + 1):
        img = images[i - 1]
        a, b = split_captcha(img)
        captcha_transformation.v1.split_captcha.example.save_captcha(img[:a, :img.shape[1]], i)
        datasets_split_captcha.append(img[:a - 1, :img.shape[1]])
        img1 = captcha_transformation.v1.split_captcha.split_captcha.get_all_icons(image=img, row=a, column=b)
        icons = captcha_transformation.v1.split_captcha.split_captcha.split_icons(img1)
        ans = captcha_transformation.v1.split_captcha.split_captcha.create_icons(image=img1, icons=icons)
        for j in range(1, len(ans) + 1):
            captcha_transformation.v1.split_captcha.example.save_icon(index_icon=j, index_image=i, icon=ans[j - 1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    res = []
    step = 1
    tol = 10
    count_contour = 1000
    for img in datasets_split_captcha:
        if not (img is None):
            res.append(discolor(image=img, step=step, tol=tol, count_contour=count_contour, new_tint_icon=Color.WHITE,
                                blur=True))
    if len(res) > 0:
        save_result(res=res, step=step, tol=tol, count_contour=count_contour)
