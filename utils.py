import numpy as np
import math


def euclidean_distance(a, b):
    dx = (a[0] - b[0]) ** 2
    dy = (a[1] - b[1]) ** 2
    return math.sqrt(dx + dy)


def angle(a, b, c):
    p = (
        euclidean_distance(a, c) ** 2
        + euclidean_distance(a, b) ** 2
        - euclidean_distance(b, c) ** 2
    )
    q = 2 * euclidean_distance(a, c) * euclidean_distance(a, b)
    return math.acos(int(p) / int(q))


degrees = lambda r: math.degrees(r) + (360 if r < 0 else 0)


def sort_counterclockwise(points, centre):

    centre_y, centre_x = centre
    angles = [degrees(np.arctan2(y - centre_y, x - centre_x)) for y, x in points]
    counterclockwise_indices = sorted(range(len(points)), key=lambda i: angles[i])
    if any(
        abs(
            angles[counterclockwise_indices[i]]
            - angles[counterclockwise_indices[i + 1]]
        )
        < 18
        for i in range(len(counterclockwise_indices) - 1)
    ):
        return 0

    counterclockwise_points = [points[i] for i in counterclockwise_indices]
    return counterclockwise_points
