
def eq_color(color_1, color_2):
    return (
            color_1[0] == color_2[0] and
            color_1[1] == color_2[1] and
            color_1[2] == color_2[2]
    )


def around_black(color_1, tol=50):
    return (
            0 <= color_1[0] <= tol and
            0 <= color_1[1] <= tol and
            0 <= color_1[2] <= tol
    )


def around_white(color_1):
    return (
            240 <= color_1[0] <= 255 and
            240 <= color_1[1] <= 255 and
            240 <= color_1[2] <= 255
    )


def minus_color(color_1, color_2):
    return (
        color_1[0] - color_2[0],
        color_1[1] - color_2[1],
        color_1[2] - color_2[2]
    )


def abs_minus_color(color_1, color_2):
    return (
        abs(color_1[0] - color_2[0]),
        abs(color_1[1] - color_2[1]),
        abs(color_1[2] - color_2[2])
    )

