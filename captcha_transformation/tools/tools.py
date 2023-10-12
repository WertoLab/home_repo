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
