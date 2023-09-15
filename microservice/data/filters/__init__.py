class DiscolorFilter:
    def __init__(self, step, tolerance, count_contour, blur):
        self.step = step
        self.tolerance = tolerance
        self.count_contour = count_contour
        self.blur = blur


class RequestDiscolor:
    def __init__(self, screenshot_captcha, screenshot_icons, filter: DiscolorFilter):
        self.screenshot_captcha = screenshot_captcha
        self.screenshot_icons = screenshot_icons
        self.filter = filter

    @classmethod
    def fromJson(cls, json: dict):
        return cls(
            screenshot_captcha=json.get("screenshot_captcha"),
            screenshot_icons=json.get("screenshot_icons"),
            filter=DiscolorFilter(step=json.get("filter").get("step"),
                                  count_contour=json.get("filter").get("count_contour"),
                                  tolerance=json.get("filter").get("tolerance"),
                                  blur=json.get("filter").get("count_contour"))
        )


class SobelFilter:
    def __init__(self, value):
        self.value = value


class RequestSobel:
    def __init__(self, screenshot_captcha, screenshot_icons, filter: SobelFilter):
        self.screenshot_captcha = screenshot_captcha
        self.screenshot_icons = screenshot_icons
        self.filter = filter

    @classmethod
    def fromJson(cls, json):
        return cls(
            screenshot_captcha=json.get("screenshot_captcha"),
            screenshot_icons=json.get("screenshot_icons"),
            filter=SobelFilter(json.get("filter").get("value"))
        )


class RequestImagesOnly:
    def __init__(self, screenshot_captcha, screenshot_icons):
        self.screenshot_captcha = screenshot_captcha
        self.screenshot_icons = screenshot_icons

    @classmethod
    def fromJson(cls, json: dict):
        return cls(
            screenshot_captcha=json.get("screenshot_captcha"),
            screenshot_icons=json.get("screenshot_icons"),
        )
