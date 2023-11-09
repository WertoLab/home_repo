from microservice.controller import init_routes
from microservice.service import Service

from flask import Flask

app = Flask(__name__)

service = Service()
init_routes(app, service)
