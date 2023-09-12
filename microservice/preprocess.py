from captcha_transformation.v2.discolor_captcha.discolor_captcha import discolor
from captcha_transformation.v2.split_captcha.split_captcha import split_icons,create_icons
from captcha_transformation.values import Color
from microservice.data.filters import DiscolorFilter


def preprocess_captcha_v2_business(img, icons, filter:DiscolorFilter):
    return_img = discolor(image=img, step=filter.step,
                          tol=filter.tolerance,
                          count_contour=filter.count_contour, new_tint_icon=Color.WHITE,
                          blur=filter.blur)
    icons_d = split_icons(icons)
    ans = create_icons(image=icons, icons=icons_d)

    return return_img, ans


def preprocess_captcha_v2(img, icons):
    return_img = discolor(image=img, step=2,
                          tol=6,
                          count_contour=1400, new_tint_icon=Color.WHITE,
                          blur=True)
    icons_d = split_icons(icons)
    ans = create_icons(image=icons, icons=icons_d)

    return return_img, ans
