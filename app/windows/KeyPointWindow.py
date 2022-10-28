from cv2 import cv2
import numpy as np

from app.constants import *
from app.windows import BaseWindow

from utils.drawing import draw_point
from utils.analysis import distance_between_points


class KeyPointWindow(BaseWindow):
    _object_correction_distance = CURSOR_SIZE * 2

    def __init__(self, path):
        super(KeyPointWindow, self).__init__(path)
        self._lbutton_pressed = False
        self.points = []
        self.point_index = None
        self.x = 0
        self.y = 0

        # Инициализация окна и виджетов
        if type(self) is KeyPointWindow:
            self._init_window()
            self._init_widgets()

    def _init_widgets(self):
        """Инициализирует виджеты."""
        super(KeyPointWindow, self)._init_widgets()
        cv2.setMouseCallback(self.name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, *userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._lbutton_pressed = True
            if self.point_index is None:  # Если курсор не находится над точкой
                self._put_point()
        if event == cv2.EVENT_LBUTTONUP:
            self._lbutton_pressed = False
        if event == cv2.EVENT_RBUTTONDOWN:
            self._remove_point()
        if event == cv2.EVENT_MOUSEMOVE:
            if self._lbutton_pressed and self.point_index is not None:  # Если нажата
                delta_x = self.x - x
                delta_y = self.y - y
                self.points[self.point_index] = (self.points[self.point_index][0] - delta_x,
                                                 self.points[self.point_index][1] - delta_y)
            self.x = x
            self.y = y
            self._update_image()

    def _update_image(self):
        image = self.image.copy()
        mouse = (self.x, self.y)  # Координаты мыши
        self.point_index, distance = self._find_closest_object(return_distance=True)  # Находим ближайший отрезок
        if distance > self._object_correction_distance:  # Если находится на двойном расстоянии курсора
            self.point_index = None  # Курсор не находится ни над какой точкой
        for i, point in enumerate(self.points):
            # Если на точку наведён курсор, то её цвет - белый, иначе - зелёный
            color = COLOR_GREEN if i != self.point_index else COLOR_WHITE
            image = draw_point(image, point, color)
        # Отрисуем курсор
        image = cv2.circle(image, mouse, CURSOR_SIZE, COLOR_WHITE, thickness=1)
        cv2.imshow(self.name, image)

    def _put_point(self):
        self.points.append((self.x, self.y))
        self._update_image()

    def _remove_point(self):
        image = self.image.copy()
        if len(self.points) == 0:  # Если точек нет
            return
        elif self.point_index is not None:
            self.points.pop(self.point_index)  # Удаляем
        cv2.imshow(self.name, image)

    def _find_closest_object(self, return_distance=False):
        closest_line = None
        min_distance = np.inf
        # Перебираем все отрезки
        for i, point in enumerate(self.points):
            distance = distance_between_points(point, (self.x, self.y))
            # Обновляем ближайший объект
            if distance < min_distance:
                min_distance = distance
                closest_line = i

        if return_distance:
            return closest_line, min_distance
        return closest_line
