import numpy as np
from captcha_transformation.values import Color


def split_icons(image, max_icon_width=90):
    """
        Функция для разбиения изображения на набор отдельных иконок.

        Args:
            image: Изображение в формате NumPy.
            max_icon_width: Максимальная ширина отдельной иконки.

        Returns:
            Список наборов пикселей, представляющих собой отдельные иконки.

        Description:

        Функция `split_icons()` разбивает изображение на набор отдельных иконок, используя порог яркости 127.
        Функция ищет горизонтальные линии, разделяющие иконки, и сохраняет наборы пикселей каждой иконки.

        Restriction:

        Функция не требует наличия каких-либо дополнительных библиотек или моделей.

        Notes:

        Функция использует порог яркости 127 для определения границ иконок. Значение порога может быть изменено в
        зависимости от особенностей изображения.
    """

    def around_black(color, tol=50):
        # Векторизованная проверка черного цвета вокруг пикселя
        return np.any(color <= tol)

    num_rows, num_cols, _ = image.shape
    icons = []
    col = 0

    while col < num_cols:
        # Ищем столбцы, которые содержат черные пиксели
        col_has_black = np.array([around_black(image[row, col]) for row in range(num_rows)])
        if np.any(col_has_black):
            next_col = col
            while next_col < num_cols and np.any([around_black(image[row, next_col]) for row in range(num_rows)]):
                next_col += 1
            # Проверяем ширину иконки и сливаем с предыдущей, если это необходимо
            icon_width = next_col - col
            if icon_width <= 55 and icons:
                if len(icons[-1]) + icon_width <= max_icon_width + 1:
                    icons[-1].extend(range(col, next_col))
                else:
                    icons.append(list(range(col, next_col)))
            elif icon_width > 0:
                icons.append(list(range(col, next_col)))
            col = next_col
        else:
            col += 1
    return icons


def create_icons(image, icons):
    """
        Функция для создания наборов отдельных иконок из одного изображения.

        Args:
            image: Изображение в формате NumPy.
            icons: Список наборов пикселей, представляющих собой отдельные иконки.

        Returns:
            Список отдельных иконок в формате NumPy.
    """
    ans = []
    for icon_cols in icons:
        start_col, end_col = icon_cols[0], icon_cols[-1]
        icon_image = image[:, start_col:end_col + 1]
        ans.append(icon_image)
    return ans


