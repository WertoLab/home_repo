FROM python:3.10

RUN mkdir /captcha_solver_app

WORKDIR /captcha_solver_app

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . .

CMD gunicorn controller:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind=0.0.0.0:8000 --timeout 600