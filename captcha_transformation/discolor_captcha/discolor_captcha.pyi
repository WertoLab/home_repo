import numpy

from captcha_transformation.contour.contour import Pixel


def discolor(image: numpy.ndarray[numpy.ndarray[tuple[int, int, int]]], step: int, tol: int, blur: bool,
             count_contour: int,
             new_tint_icon: tuple[int, int, int]) -> numpy.ndarray[numpy.ndarray[tuple[int, int, int]]]: ...


def get_neighbors(image: numpy.ndarray[numpy.ndarray[tuple[int, int, int]]], x: int, y: int, tol: int, color: \
        tuple[int, int, int]) -> list[tuple[int, int]]: ...


def create_image_from_pixels(image: numpy.ndarray[numpy.ndarray[tuple[int, int, int]]], pixel_list: list[Pixel],
                             tint: tuple[int, int, int] = None) -> numpy.ndarray[
    numpy.ndarray[tuple[int, int, int]]]: ...
