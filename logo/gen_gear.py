import math
from typing import TypeAlias

Vec2: TypeAlias = tuple[float, float]

# #### Settings for small rush logo #### #
RADIUS: int = 90
INNER_RADIUS: int = 45
CENTER: Vec2 = (100.0, 100.0)
ROTATION: float = 216
WEDGE_COUNT: int = 10
# between 0 and 1
WEDGE_DEPTH: float = 0.6
# smaller values cause smaller spikes
WIDTH_DIFF_FACTOR: float = 4.5
CUTOFF_AFTER: int = 4

# # #### Settings for big rush logo #### #
# RADIUS: int = 90
# INNER_RADIUS: int = 45
# CENTER: Vec2 = (100.0, 100.0)
# ROTATION: float = 0
# WEDGE_COUNT: int = 10
# # between 0 and 1
# WEDGE_DEPTH: float = 0.6
# # smaller values cause smaller spikes
# WIDTH_DIFF_FACTOR: float = 4.5
# CUTOFF_AFTER: int = 0


def rotate_vec(vec: Vec2, angle: float, origin: Vec2 = CENTER) -> Vec2:
    radians = math.radians(angle)
    translated = (vec[0] - origin[0], vec[1] - origin[1])
    rotated = (
        math.cos(radians) * translated[0] - math.sin(radians) * translated[1],
        math.sin(radians) * translated[0] + math.cos(radians) * translated[1],
    )
    return (rotated[0] + origin[0], rotated[1] + origin[1])


def add_vecs(vec1: Vec2, vec2: Vec2) -> Vec2:
    return (vec1[0] + vec2[0], vec1[1] + vec2[1])


def sub_vecs(vec1: Vec2, vec2: Vec2) -> Vec2:
    return (vec1[0] - vec2[0], vec1[1] - vec2[1])


def scale_vec(vec: Vec2, factor: float) -> Vec2:
    return (vec[0] * factor, vec[1] * factor)


def half_point(vec1: Vec2, vec2: Vec2) -> Vec2:
    return scale_vec(add_vecs(vec1, vec2), 0.5)


def half_quadratic(
    curve: tuple[Vec2, Vec2, Vec2]
) -> tuple[Vec2, Vec2, Vec2, Vec2]:
    curve_as_cubic = (
        curve[0],
        add_vecs(curve[0], scale_vec(sub_vecs(curve[1], curve[0]), 2.0 / 3.0)),
        add_vecs(curve[2], scale_vec(sub_vecs(curve[1], curve[2]), 2.0 / 3.0)),
        curve[2],
    )

    a = curve_as_cubic[0]
    b = curve_as_cubic[1]
    c = curve_as_cubic[2]
    d = curve_as_cubic[3]

    e = half_point(a, b)
    f = half_point(b, c)
    g = half_point(c, d)

    h = half_point(e, f)
    j = half_point(f, g)

    k = half_point(h, j)

    return (a, e, h, k)


def calc_curve_point(point1: Vec2, point2: Vec2) -> Vec2:
    middle_point = half_point(point1, point2)
    dir_vec = sub_vecs(middle_point, CENTER)
    return (
        CENTER[0] + WEDGE_DEPTH * dir_vec[0],
        CENTER[1] + WEDGE_DEPTH * dir_vec[1],
    )


def main():
    # calculate corner points
    points: list[Vec2] = []
    delta = 360.0 / (WEDGE_COUNT * 2)
    delta_spike = delta - (delta / WIDTH_DIFF_FACTOR)
    delta_wedge = delta + (delta / WIDTH_DIFF_FACTOR)
    angle = delta_wedge / 2
    base_vec: Vec2 = (CENTER[0], CENTER[1] - RADIUS)
    while angle < 360:
        points.append(rotate_vec(base_vec, angle + ROTATION))
        angle += delta_spike
        points.append(rotate_vec(base_vec, angle + ROTATION))
        angle += delta_wedge

    did_cutoff = False
    # move to start point
    path = f'M {points[0][0]} {points[0][1]}'

    for i in range(1, math.ceil(len(points))):
        point1 = points[(2 * i - 1) % len(points)]
        point2 = points[(2 * i) % len(points)]

        # arc
        path += f' A {RADIUS} {RADIUS} 0 0 1 {point1[0]} {point1[1]}'

        curve_point = calc_curve_point(point1, point2)
        if i == CUTOFF_AFTER:
            # half wedge at end
            curve = half_quadratic((point1, curve_point, point2))
            path += (
                f' C {curve[1][0]} {curve[1][1]}'
                + f' {curve[2][0]} {curve[2][1]}'
                + f' {curve[3][0]} {curve[3][1]}'
            )
            did_cutoff = True
            break
        # wedge
        path += f' Q {curve_point[0]} {curve_point[1]} {point2[0]} {point2[1]}'

    if did_cutoff:
        cutoff_angle = CUTOFF_AFTER * delta_spike + CUTOFF_AFTER * delta_wedge
        inner_circle_end = rotate_vec(
            sub_vecs(CENTER, (0, INNER_RADIUS)), ROTATION
        )
        inner_circle_start = rotate_vec(inner_circle_end, cutoff_angle)
        # line to circle start
        path += f' L {inner_circle_start[0]} {inner_circle_start[1]}'
        # inner circle
        path += (
            f' A {INNER_RADIUS} {INNER_RADIUS}'
            + f' 0 {int(cutoff_angle > 180)} 0'
            + f' {inner_circle_end[0]} {inner_circle_end[1]}'
        )

        curve = half_quadratic(
            (
                points[0],
                calc_curve_point(points[0], points[-1]),
                points[-1],
            )
        )
        # line from circle end
        path += f' L {curve[3][0]} {curve[3][1]}'
        # half wedge at beginning
        path += (
            f' C {curve[2][0]} {curve[2][1]}'
            + f' {curve[1][0]} {curve[1][1]}'
            + f' {curve[0][0]} {curve[0][1]} Z'
        )
    else:
        # inner circle
        path += f' M {CENTER[0]} {CENTER[1] - INNER_RADIUS}'
        path += f' A 1 1 0 0 0 {CENTER[0]} {CENTER[1] + INNER_RADIUS}'
        path += f' A 1 1 0 0 0 {CENTER[0]} {CENTER[1] - INNER_RADIUS}'

    print('<svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">')
    print(f'    <path d="{path}" stroke="black" fill="lightblue" />')
    print('</svg>')


if __name__ == '__main__':
    main()
