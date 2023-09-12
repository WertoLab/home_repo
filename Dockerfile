FROM python:3.10

RUN mkdir /captcha_solver_app

WORKDIR /captcha_solver_app

COPY . /captcha_solver_app

RUN pip3 install -r requirements.txt


CMD gunicorn app:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind=0.0.0.0:5000 --timeout 6000