from cv2 import cv2
import numpy as np

from app.constants import *
from app.windows import BaseWindow

from utils.analysis import find_contours


class SegmentationWindow(BaseWindow):
    def __init__(self, path):
        super(SegmentationWindow, self).__init__(path)
        self.mask = np.zeros(self.image_shape, dtype=np.uint8)  # маска с разметкой
        self.ui_mask = np.zeros(self.image_shape, dtype=np.uint8)  # маска для отображения курсора
        self.brush_size = 5  # толщина кисти
        self._draw_flag = False  # рисуем ли
        self._erase_flag = False  # стираем ли
        self._sub_tool = 0  # Если 0, то ластик. Если 1, то анти-кисть
        self._contour_fill_type = 0  # Если 0, то контур не заполняется. Если 1, заполняется

        # Инициализация окна и виджетов
        if type(self) is SegmentationWindow:
            self._init_window()
            self._init_widgets()

    def _init_widgets(self):
        """Инициализирует виджеты."""
        super(SegmentationWindow, self)._init_widgets()
        cv2.createTrackbar("brush size", self.name, self.brush_size, 20, self._brush_size_callback)
        cv2.createTrackbar("erase/anti-paint", self.name, 0, 1, self._sub_tool_change_callback)
        cv2.createTrackbar("fill contour/not", self.name, 0, 1, self._contour_fill_type_change_callback)
        cv2.setMouseCallback(self.name, self._mouse_callback)

    def _brush_size_callback(self, pos):
        self.brush_size = pos

    def _sub_tool_change_callback(self, pos):
        self._sub_tool = pos

    def _contour_fill_type_change_callback(self, pos):
        self._contour_fill_type = pos

    def _update_image(self):
        """Обновляет изображение и заново его отрисовывает."""
        # Получаем маски и контуры
        positive_mask, positive_contours = self.apply_mask(self.positive_mask, color=COLOR_GREEN)
        negative_mask, negative_contours = self.apply_mask(self.negative_mask, color=COLOR_RED)
        _, cursor_contours = self.apply_mask(self.ui_mask, color=COLOR_WHITE)

        # Складываем маски и накладываем на изображение
        image = cv2.addWeighted(self.image, 0.75, positive_mask + negative_mask, 0.25, 0)

        # Рисуем поверх контуры
        image = cv2.drawContours(image, positive_contours, -1, color=COLOR_GREEN, thickness=1)
        image = cv2.drawContours(image, negative_contours, -1, color=COLOR_RED, thickness=1)
        image = cv2.drawContours(image, cursor_contours, -1, color=COLOR_WHITE, thickness=1)

        cv2.imshow(self.name, image)

    def apply_mask(self, mask, color=(255, 255, 255)):
        """Накладывает маску на изображение и возвращает его."""
        # Находим контуры маски
        contours = find_contours(mask, flag=cv2.RETR_TREE if self._contour_fill_type == 0 else cv2.RETR_EXTERNAL)

        # Накладываем маску и её контуры
        viz = np.zeros(self.image.shape, dtype=np.uint8)
        viz = cv2.drawContours(viz, contours, -1, color=color, thickness=-1)

        return viz, contours

    def _draw_circle(self, x, y):
        if self._draw_flag:
            cv2.circle(self.mask, (x, y), self.brush_size, COLOR_POSITIVE, -1)
        elif self._erase_flag and self._sub_tool == 0:
            cv2.circle(self.mask, (x, y), self.brush_size, COLOR_EMPTY, -1)
        elif self._erase_flag and self._sub_tool == 1:
            cv2.circle(self.mask, (x, y), self.brush_size, COLOR_NEGATIVE, -1)
        self.ui_mask = np.zeros(self.image_shape, dtype=np.uint8)  # маска для отображения курсора
        cv2.circle(self.ui_mask, (x, y), self.brush_size, 1, 2)
        self._update_image()

    def _mouse_callback(self, event, x, y, flags, *userdata):
        """Обработка событий мыши."""
        super(SegmentationWindow, self)._mouse_callback(event, x, y, flags, *userdata)
        # Рисование
        if event == cv2.EVENT_LBUTTONDOWN:  # Если нажали ЛКМ
            self._draw_flag = True  # Теперь мы рисуем
            self._erase_flag = False  # и не стираем
            self._draw_circle(x, y)
        if event == cv2.EVENT_LBUTTONUP:
            self._draw_flag = False  # Теперь мы не рисуем

        # Стирание
        if event == cv2.EVENT_RBUTTONDOWN:  # Если нажали ПКМ
            self._draw_flag = False  # Теперь мы стираем
            self._erase_flag = True  # и не рисуем
            self._draw_circle(x, y)
        if event == cv2.EVENT_RBUTTONUP:
            self._erase_flag = False  # Теперь мы не стираем

        # Передвижение мыши
        if event == cv2.EVENT_MOUSEMOVE:
            self._draw_circle(x, y)

    @property
    def positive_mask(self):
        return np.array(self.mask == 1, dtype=np.uint8)

    @property
    def negative_mask(self):
        return np.array(self.mask == 2, dtype=np.uint8)
