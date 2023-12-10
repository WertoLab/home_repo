from fastapi import FastAPI

from captcha_resolver.service import Service
from captcha_resolver.controller import init_routes

app = FastAPI()
service = Service()

init_routes(app, service)

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