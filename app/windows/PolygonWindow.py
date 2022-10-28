from cv2 import cv2
import numpy as np

from app.constants import *
from app.windows import BaseWindow

from utils.drawing import draw_point, draw_line, draw_text
from utils.analysis import distance_between_points


class PolygonWindow(BaseWindow):
    _object_correction_distance = CURSOR_SIZE * 3

    def __init__(self, path):
        super(PolygonWindow, self).__init__(path)
        self._lbutton_pressed = False
        self._point_index = None
        self._polygon_index = None
        self._polygon_creating = False
        self.polygons = [[]]
        self.x = 0
        self.y = 0

        # Инициализация окна и виджетов
        if type(self) is PolygonWindow:
            self._init_window()
            self._init_widgets()

    def _init_widgets(self):
        """Инициализирует виджеты."""
        super(PolygonWindow, self)._init_widgets()
        cv2.setMouseCallback(self.name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, *userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._lbutton_pressed = True
            self._put_point()
        if event == cv2.EVENT_LBUTTONUP:
            self._lbutton_pressed = False
        if event == cv2.EVENT_RBUTTONDOWN:
            self._remove_point()
        if event == cv2.EVENT_MOUSEMOVE:
            if self._lbutton_pressed and self._polygon_index is not None:
                if self._point_index == len(self.polygons[self._polygon_index]) - 1 or self._point_index == 0:
                    self._point_index = 0
                    self._move_point(self.x - x, self.y - y)
                    self._point_index = len(self.polygons[self._polygon_index]) - 1
                self._move_point(self.x - x, self.y - y)
            self.x = x
            self.y = y
            self._update_image()

    def _move_point(self, delta_x, delta_y):
        point = self.polygons[self._polygon_index][self._point_index]
        self.polygons[self._polygon_index][self._point_index] = (point[0] - delta_x, point[1] - delta_y)

    def _update_image(self):
        image = self.image.copy()
        mouse = (self.x, self.y)  # Координаты мыши

        if not self._lbutton_pressed:
            self._polygon_index, self._point_index, distance = self._find_closest_object(
                return_distance=True)  # Находим ближайший отрезок
            if distance > self._object_correction_distance:  # Если находится на двойном расстоянии курсора
                self._polygon_index = None
        for i, polygon in enumerate(self.polygons):  # Правильно отрисуем каждый отрезок
            # Если отрезок ещё не установлен, то вторая координата - мышь
            if i == len(self.polygons) - 1:
                polygon = polygon + [mouse]
            # Если на отрезок наведён курсор, то её цвет - белый, иначе - зелёный
            color = COLOR_GREEN if i != self._polygon_index else COLOR_WHITE

            # Отрисовка
            prev_point = polygon[0]
            image = draw_point(image, prev_point, color)
            for point in polygon[1:]:
                image = draw_line(image, [prev_point, point], color)
                image = draw_point(image, point, color)
                prev_point = point

        # Отрисуем курсор
        image = draw_point(image, mouse, color=COLOR_WHITE)
        cv2.imshow(self.name, image)

    def _put_point(self):
        if self._polygon_index is not None:
            if self._point_index == 0:
                self.polygons[-1].append(self.polygons[-1][0])
                self._point_index = None
                self.polygons.append([])
        else:
            self.polygons[-1].append((self.x, self.y))  # Создаём новый
        self._update_image()

    def _remove_point(self):
        if self._polygon_index is None:
            return
        self.polygons[self._polygon_index].pop(self._point_index)
        self._update_image()

    def _find_closest_object(self, return_distance=False):
        closest_polygon = None
        closest_point = None
        min_distance = np.inf

        # Перебираем все отрезки
        for i, polygon in enumerate(self.polygons):
            if len(polygon) == 1:
                continue
            for j, point in enumerate(polygon):
                distance = distance_between_points(point, (self.x, self.y))
                # Обновляем ближайшую точку
                if distance < min_distance:
                    min_distance = distance
                    closest_polygon = i
                    closest_point = j

        if return_distance:
            return closest_polygon, closest_point, min_distance
        return closest_polygon, closest_point
