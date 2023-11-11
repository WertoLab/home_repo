from captcha_transformation.split_captcha.split_captcha import split_icons,create_icons

def preprocess_captcha_sobel(icons):
    icons_d = split_icons(icons)
    ans = create_icons(image=icons, icons=icons_d)

    return ans
