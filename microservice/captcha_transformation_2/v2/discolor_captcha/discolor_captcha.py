import uuid
from collections import deque
import cv2
import numpy as np
from captcha_transformation_2.v2.contour.contour import Contour, Pixel
from captcha_transformation.values import Color


def discolor(image, step, tol, blur=False, count_contour=100, new_tint_icon=Color.WHITE):
    if blur:
        img = cv2.GaussianBlur(image, (1, 1), 0)
    else:
        img = np.array(image)
    img_data = [
        [
            [*img[i][j], 0] for j in range(img.shape[1])
        ] for i in range(img.shape[0])
    ]
    img_data = np.array(img_data)
    # #shape[0] - rows, shape [1]- colums , shape[2] - content
    contours = []
    contour = Contour(iid=uuid.uuid4())
    for i in range(0, img.shape[0] - step, step + 1):
        for j in range(0, img.shape[1] - step, step + 1):
            if img_data[i][j][3] == 0:
                queue = deque([(i, j)])
                img_data[i][j][3] = 1
                while queue:
                    i_row, i_col = queue.popleft()
                    contour.add_pixel(
                        Pixel
                        (index_row=i_row, index_column=i_col,
                         color=img[i_row][i_col], iid=contour.uuid),
                    )
                    neighbors = get_neighbors(image=img_data, row=i_row, column=i_col, color=contour.color, tol=tol)
                    for nx, ny in neighbors:
                        img_data[nx][ny][3] = 1
                        queue.append((nx, ny))
                contours.append(contour)
                contour = Contour(iid=uuid.uuid4())
    contours = sorted(contours, key=lambda x: x.size)
    contours.reverse()
    image = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 0
    for i in range(0, min(len(contours), count_contour)):
        image = create_image_from_pixels(pixel_list=contours[i].pixel_list, image=image, tint=new_tint_icon)
    return image


def get_neighbors(row, column, image, tol, color):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
                  (1, 1), (-1, -1), (1, -1), (-1, 1)]
    neighbors = []
    for dx, dy in directions:
        i_row, i_col = row + dx, column + dy
        if 0 <= i_row < len(image) and 0 <= i_col < len(image[0]):
            if (image[i_row][i_col][3] == 0 and
                    abs(int(image[i_row][i_col][0]) - int(color[0])) <= tol
                    and abs(int(image[i_row][i_col][1]) - int(color[1])) <= tol
                    and abs(int(image[i_row][i_col][2]) - int(color[2])) <= tol):
                neighbors.append((i_row, i_col))
    return neighbors


def create_image_from_pixels(pixel_list, image, tint=None):
    new_image = image
    for pixel in pixel_list:
        if tint is None:
            new_image[pixel.row, pixel.column] = pixel.color
        else:
            new_image[pixel.row, pixel.column] = tint
    return new_image
