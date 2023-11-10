from captcha_report.database.models import Base
from captcha_report.config.di_container import setup_di_container
from captcha_report.api.captcha_report import router

setup_di_container()
