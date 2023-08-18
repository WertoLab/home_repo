import numpy


"""
Функция возвращает номер строчки  и  максимальное  кол-во найденных на ней белых пикселей с помощью  `tools.around_white()` ,
но не меньше 300 идущих друг за другом.  
Параметры:
 обязательные:
    image - оригинальная картинка с капчей
"""
def split_captcha(image: numpy.ndarray[numpy.ndarray[tuple[int, int, int]]]) -> tuple[int, int]: ...

"""
Функция возвращает картинку  с ппиктограммами.
Параметры:
 обязательные:
    image - оригинальная картинка с капчей
 необязательные:
    row - строчка с какой начинаем искать картинку  с иконками
    column - колонка до какой ищем картинку  с иконками
"""
def get_all_icons(image: numpy.ndarray[numpy.ndarray[tuple[int, int, int]]],row:int=None,column:int=None) ->\
        numpy.ndarray[numpy.ndarray[tuple[int, int, int]]]: ...


"""
Функция возвращает  список, в котором список столбцов, в которых был найден черный пиксель, с помощью `tools.around_black()`, что означает иконку. 
Параметры:
 обязательные:
    image - картинка с иконками из `get_all_icons`
 необязательные:
    max_icon_width - максимальная ширина иконки 
"""
def split_icons(image: numpy.ndarray[numpy.ndarray[tuple[int, int, int]]], max_icon_width:int=21) -> \
        list[list[int]]: ...


"""
Функция возвращает  список картинкок (иконок)
Параметры:
 обязательные:
   image - картинка с иконками из `get_all_icons`
   icons- картинка с иконками из `get_all_icons`
"""
def create_icons( image:numpy.ndarray[numpy.ndarray[tuple[int, int, int]]],icons:list[list[int]]) ->numpy.ndarray[numpy.ndarray[numpy.ndarray[tuple[int, int, int]]]]: ...
