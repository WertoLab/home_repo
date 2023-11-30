from captcha_resolver.controller import init_routes
from captcha_resolver.service import Service
from gunicorn.app.base import BaseApplication
from fastapi import FastAPI

app = FastAPI()

service = Service()
init_routes(app, service)


class GunicornApp(BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        config = {key: value for key, value in self.options.items() if key in self.cfg.settings and value is not None}
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application

'''
if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=8000)

    options = {
        'bind': '0.0.0.0:8000',
        'workers': 4,
        'worker_class': 'uvicorn.workers.UvicornWorker',
        'timeout': 60,
    }
    g = GunicornApp(app, options).run()
'''