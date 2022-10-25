def distance_between_points(p1, p2, spacing=(1, 1)):
    return (((p1[0] - p2[0]) * spacing[1]) ** 2 + ((p1[1] - p2[1]) * spacing[0]) ** 2) ** 0.5


def in_stripe(point, func1, func2, hard=False):
    x, y = point
    min_val, max_val = sorted([func1(x), func2(x)])
    if hard:
        return min_val < y < max_val
    return min_val <= y <= max_val
