class SobelFilter:
    def __init__(self, value):
        self.value = value


class RequestBusiness:
    def __init__(self, screenshot_captcha, screenshot_icons, filter):
        self.filter = filter
        self.screenshot_captcha = screenshot_captcha
        self.screenshot_icons = screenshot_icons

    @classmethod
    def fromJson(cls, json: dict):
        return cls(
            screenshot_captcha=json.get("body"),
            screenshot_icons=json.get("imginstructions"),
            filter=json.get("sobel_filter")
        )
