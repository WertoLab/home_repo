

"""
Функция возвращает true or false . В зависимости от того близок ли цвет к черному с заданной погрешностью
Параметры:
 обязательные:
    color_1 - (R:int,G:int,B:int)
"""
def around_black(color_1: tuple[int, int, int]) -> bool: ...

"""
Функция возвращает true or false . В зависимости от того близок ли цвет к белому с заданной погрешностью
Параметры:
 обязательные:
    color_1 - (R:int,G:int,B:int)
"""
def around_white(color_1: tuple[int, int, int]) -> bool: ...

