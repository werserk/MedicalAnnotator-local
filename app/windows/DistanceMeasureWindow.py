from cv2 import cv2
import numpy as np

from app.constants import *
from app.windows import BaseWindow

from utils.drawing import draw_dot, draw_line, draw_text
from utils.analysis import distance_between_points, in_stripe


class DistanceMeasureWindow(BaseWindow):
    _line_correction_distance = CURSOR_SIZE * 2

    def __init__(self, path):
        super(DistanceMeasureWindow, self).__init__(path)
        self._lbutton_pressed = False
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
            self._put_dot()
        if event == cv2.EVENT_LBUTTONUP:
            self._lbutton_pressed = False
        if event == cv2.EVENT_RBUTTONDOWN:
            self._remove_dot()
        if event == cv2.EVENT_MOUSEMOVE:
            self.x = x
            self.y = y
            self._update_image()

    def _update_image(self):
        image = self.image.copy()
        mouse = (self.x, self.y)

        line_index, distance = self._find_closest_line(return_distance=True)  # Находим ближайший отрезок
        if distance > self._line_correction_distance:  # Если находится на двойном расстоянии курсора
            line_index = -1
        for i, line in enumerate(self.lines):
            line = [line[0], mouse] if len(line) == 1 else line
            color = COLOR_GREEN if i != line_index else COLOR_WHITE
            image = draw_line(image, line, color)
            image = draw_dot(image, line[0], color)
            image = draw_dot(image, line[1], color)
            distance = distance_between_points(*line, spacing=self.spacing)
            image = draw_text(image, str(round(distance, 2)) + 'mm', line[1], text_color_bg=color)
        image = draw_dot(image, mouse, color=COLOR_WHITE)
        cv2.imshow(self.name, image)

    def _put_dot(self):
        if len(self.lines) == 0 or len(self.lines[-1]) == 2:  # Если отрезков нет или они все установлены
            self.lines.append([(self.x, self.y)])  # Создаём новый
        else:
            self.lines[-1].append((self.x, self.y))  # Иначе добавляем конец отрезка
        self._update_image()

    def _remove_dot(self):
        image = self.image.copy()
        if len(self.lines) == 0:  # Если точек нет
            return
        last_line = self.lines[-1]  # Получаем последний уставленный отрезок
        if len(last_line) == 1:  # Если этот отрезок ещё в процессе установки
            self.lines.pop(-1)  # То отменяем его установку
        else:  # Если все отрезки установлены
            line_index, distance = self._find_closest_line(return_distance=True)  # Находим ближайший отрезок
            if distance < self._line_correction_distance:  # Если находится на двойном расстоянии курсора
                self.lines.pop(line_index)  # Удаляем
        cv2.imshow(self.name, image)

    def _find_closest_line(self, return_distance=False):
        closest_line = None
        min_distance = np.inf
        # Перебираем все отрезки
        for i, line in enumerate(self.lines):
            if len(line) == 1:
                continue
            (x1, y1), (x2, y2) = line
            # Получаем коэффициент при X (если вертикальная линия, то None)
            b = (y1 - y2) / (x1 - x2) if (x1 - x2) != 0 else None
            if b == 0:
                if return_distance:
                    return None, np.inf
                return None
            else:
                # Получаем длины сторон треугольника, образуемые концами отрезка и мышкой
                distance0 = distance_between_points((x1, y1), (x2, y2))
                distance1 = distance_between_points((x1, y1), (self.x, self.y))
                distance2 = distance_between_points((x2, y2), (self.x, self.y))
                if distance1 ** 2 + distance2 ** 2 >= distance0 ** 2:  # Если треугольник тупоугольный
                    distance = min(distance1, distance2)  # То берём расстояние до одной из точек (минимальное)
                else:  # Иначе считаем от мышки расстояние до прямой
                    if b is not None:  # Если не вертикальная линия
                        c = y1 - b * x1  # Получаем свободный коэффициент
                        distance = abs(self.y - b * self.x - c) / (b ** 2 + 1) ** 0.5
                    else:
                        distance = abs(self.x - x1)

            # Обновляем ближайший отрезок
            if distance < min_distance:
                min_distance = distance
                closest_line = i

        if return_distance:
            return closest_line, min_distance
        return closest_line
