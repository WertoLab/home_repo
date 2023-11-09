FROM python:3.10

WORKDIR /captcha_solver_app

COPY ./requirements.txt /captcha_solver_app/

RUN pip install --upgrade pip && pip install -r requirements.txt --no-cache-dir

COPY . /captcha_solver_app

EXPOSE 8000

CMD gunicorn app:app --workers 5 --worker-class gevent --bind 0.0.0.0:8000
