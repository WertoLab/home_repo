from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from microservice.controller import init_routes
from microservice.service import Service
from gunicorn.app.base import BaseApplication


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


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
    options = {
        'bind': '0.0.0.0:8000',
        'workers': 4,
        'worker_class': 'uvicorn.workers.UvicornWorker',
        'timeout': 60,
    }
    g=GunicornApp(app, options).run()















# app = Flask(__name__)
from fastapi import Response

# @app.route("/hello_world")
# def hello_world():
#     return  {"hello_world": "flask_api"}
# @app.route("/")
# def home():
#     return  {"home": "home_page"}



# @app.route("/get_captcha_solve_sequence_old_our")
# async def get_captcha_solve_sequence_old(info: str):
#     return Response(content=json.dumps(service.get_captcha_solve_sequence_old(await info.json())),
#                     media_type='application/json')


# @app.route("/get_captcha_solve_sequence_sobel_our",methods = ['GET'])
# def get_captcha_solve_sequence():
#     screenshot_captcha=request.files["screenshot_captcha"]
#     f=screenshot_captcha.stream.seek(0,2)
#     f=t
#     # print(info)
#     # json_data=json.dumps(
#     #    service.get_captcha_solve_sequence_sobel(json.loads(info))
#     # )
#


# @app.route("/get_captcha_solve_sequence_old_business")
# async def get_captcha_solve_sequence_old_business(info: str):
#     sequence, discolored, captcha, icons, answer = service.get_captcha_solve_sequence_old_business(await info.json())
#     return Response(content=json.dumps({"coordinate_sequence": sequence, "discolored_captcha": discolored, "captcha": captcha, "icons": icons, "answer": answer}),
#                     media_type='application/json')
#
#
# @app.route("/get_captcha_solve_sequence_sobel_business")
# async def get_captcha_solve_sequence_business(info: str):
#     sequence, discolored, captcha, icons, answer = service.get_captcha_solve_sequence_sobel_business(await info.json())
#     return Response(content=json.dumps({"coordinate_sequence": sequence, "discolored_captcha": discolored, "captcha": captcha, "icons": icons, "answer": answer}),
#                     media_type='application/json')

