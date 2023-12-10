FROM python:3.10.11

WORKDIR /captcha_solver_app

COPY ./requirements.txt /captcha_solver_app/

RUN pip install --upgrade pip && pip install -r requirements.txt --no-cache-dir
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY . /captcha_solver_app

EXPOSE 5000

CMD gunicorn main:app -w=1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:5000 --timeout=600
#CMD [ "gunicorn", "--bind", "0.0.0.0:8000", "main:app" ]
