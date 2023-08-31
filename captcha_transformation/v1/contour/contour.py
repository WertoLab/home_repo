import uuid


class Pixel:
    def __init__(self, color, iid: uuid.UUID, index_row: int, index_column: int):
        self.color = color
        self.uuid = iid,
        self.row = index_row
        self.column = index_column


class Contour:
    def __init__(self, iid):
        self.size = 0
        self.uuid = iid
        self.pixel_list: list[Pixel] = []
        self.color = None

    def is_contour(self):
        for j in self.pixel_list:
            pass

    def add_pixel(self, pixel: Pixel):
        if self.color is None:
            self.color = pixel.color
        self.pixel_list.append(pixel)
        self.size += 1

    def main_pixel(self):
        return self.pixel_list[0]

    def __str__(self):
        return F"Contour: size=  {self.size}\n color= {self.color}\n"
