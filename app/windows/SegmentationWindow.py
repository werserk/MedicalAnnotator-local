from cv2 import cv2
import numpy as np
from app.constants import *
from app.windows import BaseWindow


def find_exterior_contours(img):
    # Находим контуры объектов разметки
    ret = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(ret) == 2:
        return ret[0]
    if len(ret) == 3:
        return ret[1]
    raise Exception("Check the signature for `cv.findContours()`.")


class SegmentationWindow(BaseWindow):
    def __init__(self, path):
        super(SegmentationWindow, self).__init__(path)
        self.mask = np.zeros(self.image_shape, dtype=np.uint8)  # маска с разметкой
        self.brush_size = 5  # толщина кисти
        self._draw_flag = False  # рисуем ли
        self._erase_flag = False  # стираем ли

        # Инициализация окна и виджетов
        if type(self) is SegmentationWindow:
            self._init_window()
            self._init_widgets()

    def _init_widgets(self):
        """Инициализирует виджеты."""
        super(SegmentationWindow, self)._init_widgets()
        cv2.createTrackbar("brush size", self.name, self.brush_size, 20, self._brush_size_callback)
        cv2.setMouseCallback(self.name, self._mouse_callback)

    def _brush_size_callback(self, pos):
        self.brush_size = pos

    def _update_image(self):
        """Обновляет изображение и заново его отрисовывает."""
        image = self._apply_mask(self.image, self.positive_mask, color=COLOR_GREEN)
        image = self._apply_mask(image, self.negative_mask, color=COLOR_RED)
        cv2.imshow(self.name, image)

    def _apply_mask(self, image, mask, color=(255, 255, 255)):
        """Накладывает маску на изображение и возвращает его."""
        # Находим контуры маски
        contours = find_exterior_contours(mask)

        # Накладываем маску и её контуры
        viz = image.copy()
        viz = cv2.drawContours(viz, contours, -1, color=color, thickness=-1)
        viz = cv2.addWeighted(self.image, 0.75, viz, 0.25, 0)
        viz = cv2.drawContours(viz, contours, -1, color=color, thickness=1)
        return viz

    def _draw_circle(self, x, y):
        if self._draw_flag:
            cv2.circle(self.mask, (x, y), self.brush_size, COLOR_POSITIVE, -1)
        elif self._erase_flag:
            cv2.circle(self.mask, (x, y), self.brush_size, COLOR_NEGATIVE, -1)
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
        if event == cv2.EVENT_MOUSEMOVE:  # Движение мыши
            if self._draw_flag:  # Если рисуем
                self._draw_circle(x, y)
            elif self._erase_flag:  # Если стираем
                self._draw_circle(x, y)

    @property
    def positive_mask(self):
        return np.array(self.mask == 1, dtype=np.uint8)

    @property
    def negative_mask(self):
        return np.array(self.mask == 2, dtype=np.uint8)
