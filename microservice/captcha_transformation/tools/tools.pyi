

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

"""
Функция возвращает true or false . В зависимости от того равны ли значение цветов .
Параметры:
 обязательные:
    color_1,color_2 - (R:int,G:int,B:int)
"""
def eq_color(color_1: tuple[int, int, int], color_2: tuple[int, int, int])->bool:...

"""
Функция возвращает кортедж из трех  чисел типа Int . В котором поэлементная разница цветов (RGB).
R1-R2 
G1-G2
B1-B2
Параметры:
 обязательные:
    color_1,color_2 - (R:int,G:int,B:int)
"""
def minus_color(color_1: tuple[int, int, int], color_2: tuple[int, int, int]) -> tuple[int, int, int]: ...

"""
Функция возвращает кортедж из трех  чисел типа Int . В котором поэлементная по модулю разница цветов (RGB).
abs(R1-R2) 
abs(G1-G2)
abs(B1-B2)
Параметры:
 обязательные:
    color_1,color_2 - (R:int,G:int,B:int)
"""

def abs_minus_color(color_1: tuple[int, int, int], color_2: tuple[int, int, int]) -> tuple[int, int, int]: ...

