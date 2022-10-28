from cv2 import cv2
import numpy as np

from app.constants import *
from app.windows import BaseWindow

from utils.drawing import draw_point, draw_line, draw_text
from utils.analysis import distance_between_points


class DistanceMeasureWindow(BaseWindow):
    _object_correction_distance = CURSOR_SIZE * 2

    def __init__(self, path):
        super(DistanceMeasureWindow, self).__init__(path)
        self._lbutton_pressed = False
        self._point_index = None
        self._line_index = None
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
            self._lbutton_pressed = True
            if self._line_index is None:
                self._put_point()
        if event == cv2.EVENT_LBUTTONUP:
            self._lbutton_pressed = False
        if event == cv2.EVENT_RBUTTONDOWN:
            self._remove_point()
        if event == cv2.EVENT_MOUSEMOVE:
            if self._lbutton_pressed and self._line_index is not None:
                self._move_line(self.x - x, self.y - y)
            self.x = x
            self.y = y
            self._update_image()

    def _move_line(self, delta_x, delta_y):
        if self._point_index is None:
            points = self.lines[self._line_index]
            self.lines[self._line_index] = [self._move_point(points[0], delta_x, delta_y),
                                            self._move_point(points[1], delta_x, delta_y)]
        else:
            point = self.lines[self._line_index][self._point_index]
            self.lines[self._line_index][self._point_index] = self._move_point(point, delta_x, delta_y)

    def _move_point(self, point, delta_x, delta_y):
        return point[0] - delta_x, point[1] - delta_y

    def _update_image(self):
        image = self.image.copy()
        mouse = (self.x, self.y)  # Координаты мыши

        if not self._lbutton_pressed:
            self._line_index, self._point_index, distance = self._find_closest_object(
                return_distance=True)  # Находим ближайший отрезок
            if distance > self._object_correction_distance:  # Если находится на двойном расстоянии курсора
                self._line_index = None
        for i, line in enumerate(self.lines):  # Правильно отрисуем каждый отрезок
            # Если отрезок ещё не установлен, то вторая координата - мышь
            line = [line[0], mouse] if len(line) == 1 else line

            # Если на отрезок наведён курсор, то её цвет - белый, иначе - зелёный
            color = COLOR_GREEN if i != self._line_index else COLOR_WHITE

            # Отрисовка
            image = draw_line(image, line, color)
            image = draw_point(image, line[0], color)
            image = draw_point(image, line[1], color)

            # Подсчёт расстояния в миллиметрах
            distance = distance_between_points(*line, spacing=self.spacing)

            # Напишем расстояние на изображении
            image = draw_text(image, str(round(distance, 2)) + 'mm', line[1], text_color_bg=color)

        # Отрисуем курсор
        image = draw_point(image, mouse, color=COLOR_WHITE)
        cv2.imshow(self.name, image)

    def _put_point(self):
        if len(self.lines) == 0 or len(self.lines[-1]) == 2:  # Если отрезков нет или они все установлены
            self.lines.append([(self.x, self.y)])  # Создаём новый
        else:
            self.lines[-1].append((self.x, self.y))  # Иначе добавляем конец отрезка
        self._update_image()

    def _remove_point(self):
        if len(self.lines) == 0:  # Если точек нет
            return
        last_line = self.lines[-1]  # Получаем последний уставленный отрезок
        if len(last_line) == 1:  # Если этот отрезок ещё в процессе установки
            self.lines.pop(-1)  # То отменяем его установку
        else:  # Если все отрезки установлены
            if self._line_index is not None:
                if self._point_index is not None:
                    print(self._point_index)
                    self.lines.append([self.lines[self._line_index][1 - self._point_index]])
                self.lines.pop(self._line_index)  # Удаляем линию
        self._update_image()

    def _find_closest_object(self, return_distance=False):
        closest_line = None
        closest_point = None
        min_distance = np.inf

        # Перебираем все отрезки
        for i, line in enumerate(self.lines):
            if len(line) == 1:
                continue
            (x1, y1), (x2, y2) = line
            # Получаем коэффициент при X (если вертикальная линия, то None)
            b = (y1 - y2) / (x1 - x2) if (x1 - x2) != 0 else None
            if b == 0:
                continue
            else:
                # Получаем длины сторон треугольника, образуемые концами отрезка и мышкой
                distance0 = distance_between_points((x1, y1), (x2, y2))
                distance1 = distance_between_points((x1, y1), (self.x, self.y))
                distance2 = distance_between_points((x2, y2), (self.x, self.y))
                if distance1 < distance2:
                    local_min_distance = distance1
                    closest_point = 0
                else:
                    local_min_distance = distance2
                    closest_point = 1
                if distance1 ** 2 + distance2 ** 2 >= distance0 ** 2:  # Если треугольник тупоугольный
                    distance = local_min_distance  # То берём расстояние до одной из точек (минимальное)
                else:  # Иначе считаем от мышки расстояние до прямой
                    if b is not None:  # Если не вертикальная линия
                        c = y1 - b * x1  # Получаем свободный коэффициент
                        distance = abs(self.y - b * self.x - c) / (b ** 2 + 1) ** 0.5
                    else:
                        distance = abs(self.x - x1)
                    closest_point = None

            # Обновляем ближайший отрезок
            if distance < min_distance:
                min_distance = distance
                closest_line = i
                closest_point = closest_point

        if return_distance:
            return closest_line, closest_point, min_distance
        return closest_line, closest_point
