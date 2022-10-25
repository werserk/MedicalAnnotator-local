from cv2 import cv2
from app.constants import *


def draw_dot(image, coordinates, color=COLOR_GREEN):
    image = cv2.circle(image, coordinates, CURSOR_SIZE, color, thickness=-1)
    return image


def draw_line(image, line, color=COLOR_GREEN):
    image = cv2.line(image, line[0], line[1], color, thickness=1)
    return image


def distance_between_points(p1, p2, spacing=(1, 1)):
    return (((p1[0] - p2[0]) * spacing[1]) ** 2 + ((p1[1] - p2[1]) * spacing[0]) ** 2) ** 0.5
