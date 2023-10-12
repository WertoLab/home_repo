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


class RequestBusiness:
    def __init__(self, screenshot_captcha, screenshot_icons):
        self.screenshot_captcha = screenshot_captcha
        self.screenshot_icons = screenshot_icons

    @classmethod
    def fromJson(cls, json: dict):
        return cls(
            screenshot_captcha=json.get("body"),
            screenshot_icons=json.get("imginstructions"),
        )
