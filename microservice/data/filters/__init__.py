
class DiscolorFilter:
    def __init__(self, step, tolerance, count_contour,blur):
        self.step=step
        self.tolerance=tolerance
        self.count_contour=count_contour
        self.blur=blur

class RequestDiscolor:
    def __init__(self, screenshot_captcha, screenshot_icons,filter:DiscolorFilter):
        self.screenshot_captcha=screenshot_captcha
        self.screenshot_icons=screenshot_icons
        self.filter=filter

    @classmethod
    def fromJson(cls, json: dict):
        return cls(
            screenshot_captcha=json["screenshot_captcha"],
            screenshot_icons=json["screenshot_icons"],
            filter= DiscolorFilter(step=json["filter"]["step"],
                                    count_contour=json["filter"]["count_contour"],
                                   tolerance=json["filter"]["tolerance"],
                                   blur=json["filter"]["blur"])
               )
class SobelFilter:
    def __init__(self,value):
        self.value=value

class RequestSobel:
    def __init__(self, screenshot_captcha, screenshot_icons, filter: SobelFilter):
        self.screenshot_captcha = screenshot_captcha
        self.screenshot_icons = screenshot_icons
        self.filter = filter
    @classmethod
    def fromJson(cls,json:dict):
        return cls(
            screenshot_captcha=json["screenshot_captcha"],
            screenshot_icons=json["screenshot_icons"],
            filter=SobelFilter(json["filter"]["value"])
        )

class RequestImagesOnly:
    def __init__(self,screenshot_captcha,screenshot_icons):
        self.screenshot_captcha=screenshot_captcha
        self.screenshot_icons= screenshot_icons
    @classmethod
    def fromJson(cls,json:dict):
        return cls(
            screenshot_captcha=json["screenshot_captcha"],
            screenshot_icons=json["screenshot_icons"]
        )
