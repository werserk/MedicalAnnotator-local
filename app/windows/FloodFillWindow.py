from cv2 import cv2
import numpy as np
import utils.preprocessing.marking as mkg
from app.constants import *
from app.windows import SegmentationWindow
from app.windows.SegmentationWindow import find_exterior_contours


class FloodFillWindow(SegmentationWindow):
    connectivity = 4  # количество пикселей для усреднения вокруг стартового пикслея
    _flood_fill_flags = (connectivity | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | 255 << 8)

    def __init__(self, path):
        """
        Основное окно для Медицинского разметчика.
        :param path: DICOM путь для чтения и разметки
        """
        super(FloodFillWindow, self).__init__(path)

        # Читаем DICOM
        self.blurred_image = mkg.denoise(self.image, power=7)
        self.flood_mask = np.zeros(self.image_shape, dtype=np.uint8)  # маска с разметкой объекта заливкой

        # Динамические значения, которые будет изменять врач во время разметки
        self.tolerance = (10,) * 3  # максимальное отклонение значения пикселя при разметке заливкой

        self._floodfill_flag = False  # активирована ли маска заливки

        # Инициализация окна и виджетов
        if type(self) is FloodFillWindow:
            self._init_window()
            self._init_widgets()

    def _init_widgets(self):
        """Инициализирует виджеты"""
        super(FloodFillWindow, self)._init_widgets()
        # Слайдеры
        cv2.createTrackbar("Tolerance", self.name, self.tolerance[0], 127, self._tolerance_callback)

        # Обработка мыши
        cv2.setMouseCallback(self.name, self._mouse_callback)

    def _floodfill(self):
        """Разметка заливкой."""
        self.flood_mask = np.zeros(self.image_shape, dtype=np.uint8)
        all_contours = find_exterior_contours(self.positive_mask)
        # TODO: Сейчас шаг tolerance - 2, потому что floodfill, похоже, не поддерживает ничего кроме uint8
        # TODO: Необходимо избавиться от этого
        image = np.array(self.blurred_image.copy() // 2, dtype=np.uint8)
        image[self.mask == 2] = 255
        for contours in all_contours:
            for coordinates in contours:
                x, y = coordinates[0]
                _flood_mask = np.zeros((self.image_shape[0] + 2, self.image_shape[1] + 2), dtype=np.uint8)
                cv2.floodFill(image, _flood_mask, (x, y), 0, self.tolerance, self.tolerance, self._flood_fill_flags)
                flood_mask = _flood_mask[1:-1, 1:-1].copy()
                self.flood_mask = cv2.bitwise_or(self.flood_mask, flood_mask)

            centroid = contours.mean(axis=1).mean(axis=0)
            xc, yc = int(centroid[0]), int(centroid[1])
            if self.positive_mask[xc][yc] == 1:
                _flood_mask = np.zeros((self.image_shape[0] + 2, self.image_shape[1] + 2), dtype=np.uint8)
                cv2.floodFill(image, _flood_mask, (xc, yc), 0, self.tolerance, self.tolerance, self._flood_fill_flags)
                flood_mask = _flood_mask[1:-1, 1:-1].copy()
                self.flood_mask = cv2.bitwise_or(self.flood_mask, flood_mask)
        self.flood_mask = mkg.remove_small_dots(self.flood_mask)
        self._update_image()

    def _update_image(self):
        """Обновляет изображение и заново его отрисовывает."""
        mask = self.positive_mask
        if self._floodfill_flag:
            mask = cv2.bitwise_or(mask, np.array(self.flood_mask, dtype=np.uint8))
        image = self._apply_mask(self.image, mask, color=COLOR_GREEN)
        image = self._apply_mask(image, self.negative_mask, color=COLOR_RED)
        cv2.imshow(self.name, image)

    def _tolerance_callback(self, pos):
        self.tolerance = (pos,) * 3

    def _get_tolerance(self):
        return cv2.getTrackbarPos("Tolerance", self.name)

    def _set_tolerance(self, value):
        cv2.setTrackbarPos("Tolerance", self.name, min(max(0, value), 255))

    def _mouse_callback(self, event, x, y, flags, *userdata):
        """Обработка событий мыши."""
        super(FloodFillWindow, self)._mouse_callback(event, x, y, flags, *userdata)
        # Изменение коэффициента tolerance
        if event == cv2.EVENT_MOUSEWHEEL:  # Вращение колёсика мыши
            # Если колёсико мыши - вверх, то увеличиваем значение, иначе - уменьшаем
            tolerance_increment_flag = 1 if flags > 0 else -1
            self._set_tolerance(self._get_tolerance() + tolerance_increment_flag)

            # Разметка заливкой
            self._floodfill()

    def _keyboard_callback(self, key):
        """Обработка событий клавиатуры."""
        traceback = super(FloodFillWindow, self)._keyboard_callback(key)
        if traceback is not None:
            return traceback
        if key == ord("f"):
            self._floodfill_flag = not self._floodfill_flag
            print("Floodfill " + "activated" if self._floodfill_flag else "disabled")
            self._floodfill()
