import numpy
import numpy as np
from captcha_transformation.values import Color
from captcha_transformation.tools.tools import *


def split_captcha(image):
    max_k = 0
    new_image = numpy.copy(image)
    for row in range(150, new_image.shape[0]):
        k = 0
        for column in range(new_image.shape[1] - 1):
            if around_white(color_1=new_image[row][column]) and around_white(color_1=new_image[row][column + 1]):
                k += 1
                max_k = max(max_k, k)
                if 300 <= k <= new_image.shape[1]:
                    return row, column
            else:
                break


def split_icons(image, max_icon_width=21):
    next_col = 0
    icons = []
    num_rows = image.shape[0]
    num_cols = image.shape[1]
    col = 0

    while col < num_cols:
        icon_cols = []

        for row in range(num_rows):
            if around_black(image[row][col]):
                icon_cols.append(col)
                break

        if icon_cols:
            icon_width = 1
            next_col = col + 1

            while next_col < num_cols:
                all_white = True

                for row in range(num_rows):
                    if around_black(image[row][next_col]):
                        icon_cols.append(next_col)
                        icon_width += 1
                        all_white = False
                        break

                if all_white:
                    break

                next_col += 1

            if icon_width <= 15:
                if len(icons) > 0:
                    prev_icon = icons[-1]
                    if prev_icon and len(prev_icon) + icon_width <= max_icon_width + 1:
                        icons[-1].extend(range(icons[-1][-1] + 1, icon_cols[-1] + 1))
                        icon_cols = []

        if icon_cols:
            icons.append(icon_cols)

        col = max(next_col, col + 1)

    return icons


def get_all_icons(image, row=None, column=None):
    if row is None or column is None:
        row, column = split_captcha(image=image)
    return image[row:row + 12 + 20, image.shape[1] - 120:column]


def create_icons(image, icons):
    ans = []

    for i in range(0, len(icons)):
        white_matrix = np.full(shape=(image.shape[0], icons[i][-1] - icons[i][0] + 1, 3), fill_value=Color.WHITE,
                               dtype=np.uint8)
        k = 0
        for column in icons[i]:
            white_matrix[:, k] = np.copy(image[:, column])
            k += 1
        ans.append(white_matrix)
    return ans


