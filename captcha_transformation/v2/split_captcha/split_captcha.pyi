import numpy

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
