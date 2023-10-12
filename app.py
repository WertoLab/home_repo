from microservice.controller import init_routes
from microservice.service import Service
from gunicorn.app.base import BaseApplication
from flask import Flask

app = Flask(__name__)
'''
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)
'''

service = Service()
init_routes(app,service)


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


if __name__ == '__main__':
    '''
    options = {
        'bind': '0.0.0.0:8000',
        'workers': 4,
        'worker_class': 'uvicorn.workers.UvicornWorker',
        'timeout': 60,
    }
    g=GunicornApp(app, options).run()
    '''
    app.run()
