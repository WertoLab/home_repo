FROM python:3.10



RUN mkdir /captcha_solver_app

WORKDIR /captcha_solver_app

COPY . /captcha_solver_app

RUN pip3 install -r requirements.txt

EXPOSE 8000

CMD python3 app.py