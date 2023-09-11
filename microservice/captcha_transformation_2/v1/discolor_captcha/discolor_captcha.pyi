import numpy

from captcha_transformation.v1.contour.contour import Pixel
from captcha_transformation.values import Color


"""
Функция возвращает обесцвеченную капчу,
Параметры:
 обязательные:
    image -  фон капчи , если есть оригинал из  функции `split_captcha.split_captcha()`,
     можно получить  индекс  строчки, где оканчивается фон капчи (его высота).
    step - шаг с каким идем по капчи . К примеру шаг 1 - идем по каждому пикселю
    tol - допустимая разница  по модулю между цветами 
 необязательные:
    blur - нужно ли  размывать картинку
    count_contour - кол-во возвращаемых контуров 
    new_tint_icon - в какой цвет красим найденые иконки


"""


def discolor(image: numpy.ndarray[numpy.ndarray[tuple[int, int, int]]], step: int, tol: int, blur: bool = False,
             count_contour: int = 100,
             new_tint_icon: tuple[int, int, int] = Color.WHITE) -> numpy.ndarray[
    numpy.ndarray[tuple[int, int, int]]]: ...


"""
Функция возвращает  соседей пикселя  одного цвета  ( или с заданной погрешностью). Ищет в 8 направлениях.
Вызывается в функции discolor
Параметры:
 обязательные:
    image -  фон капчи , если есть оригинал из  функции `split_captcha.split_captcha()`,
     можно получить  индекс  строчки, где оканчивается фон капчи (его высота).
    row - строчка на которой находится пиксель
    column - колонка на которой находитс пиксель
    color - цвет пикселя


"""


def get_neighbors(image: numpy.ndarray[numpy.ndarray[tuple[int, int, int]]], row: int, column: int, tol: int, color: \
        tuple[int, int, int]) -> list[tuple[int, int]]: ...


"""
Функция возвращает  картинку с добавленной иконкой. Вызывается в функции discolor
Параметры:
 обязательные:
    image -  фон капчи , если есть оригинал из  функции `split_captcha.split_captcha()`,
     можно получить  индекс  строчки, где оканчивается фон капчи (его высота).
    pixel_list - список пикселей 
    tint - цвет пикселя

 необязательные:
    new_tint_icon - в какой цвет красим найденые иконки


"""


def create_image_from_pixels(image: numpy.ndarray[numpy.ndarray[tuple[int, int, int]]], pixel_list: list[Pixel],
                             tint: tuple[int, int, int] = None) -> numpy.ndarray[
    numpy.ndarray[tuple[int, int, int]]]: ...
