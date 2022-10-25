from cv2 import cv2
import numpy as np

from app.constants import *
from app.windows import BaseWindow

from utils.drawing import draw_dot, draw_line, draw_text
from utils.analysis import distance_between_points


class DistanceMeasureWindow(BaseWindow):
    def __init__(self, path):
        super(DistanceMeasureWindow, self).__init__(path)
        self.spacing = self.survey[('0028', '0030')].value
        self.lines = []
        self.x = 0
        self.y = 0

        # Инициализация окна и виджетов
        if type(self) is DistanceMeasureWindow:
            self._init_window()
            self._init_widgets()

    def _init_widgets(self):
        """Инициализирует виджеты."""
        super(DistanceMeasureWindow, self)._init_widgets()
        cv2.setMouseCallback(self.name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, *userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._put_dot()
        if event == cv2.EVENT_RBUTTONDOWN:
            self._remove_dot()
        if event == cv2.EVENT_MOUSEMOVE:
            self.x = x
            self.y = y
            self._update_image()

    def _update_image(self):
        image = self.image.copy()
        mouse = (self.x, self.y)
        for line in self.lines:
            if len(line) == 1:
                line = [line[0], mouse]
            image = draw_line(image, line)
            image = draw_dot(image, line[0])
            image = draw_dot(image, line[1])
            distance = distance_between_points(*line, spacing=self.spacing)
            image = draw_text(image, str(round(distance, 2)) + 'mm', line[1])
        image = draw_dot(image, mouse, color=COLOR_WHITE)
        cv2.imshow(self.name, image)

    def _put_dot(self):
        if len(self.lines) == 0 or len(self.lines[-1]) == 2:
            self.lines.append([(self.x, self.y)])
        else:
            self.lines[-1].append((self.x, self.y))
        self._update_image()

    def _remove_dot(self):
        image = self.image.copy()
        if len(self.lines) == 0:
            return
        last_line = self.lines[-1]
        if len(last_line) == 1:
            self.lines.pop(-1)
        else:
            line = self._find_closest_line()
            self.lines.remove(line)
        cv2.imshow(self.name, image)

    def _find_closest_line(self):
        closest_line = None
        min_distance = np.inf
        for line in self.lines:
            (x1, y1), (x2, y2) = line
            a = (y1 - y2) / (x1 - x2) if (x1 - x2) != 0 else None
            if a is not None:
                c = y1 - a * x1
                distance = abs(self.y - a * self.x - c) / (a ** 2 + 1) ** 0.5
            else:
                distance = abs(self.x - x1)
            if distance < min_distance:
                min_distance = distance
                closest_line = line
        return closest_line
