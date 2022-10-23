from cv2 import cv2
import numpy as np
import pydicom
import utils.preprocessing.dicom_transforms as dt
from .constants import *


def find_exterior_contours(img):
    # Находим контуры объектов разметки
    ret = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(ret) == 2:
        return ret[0]
    if len(ret) == 3:
        return ret[1]
    raise Exception("Check the signature for `cv.findContours()`.")


class SelectionWindow:
    name = 'Medical annotator'
    connectivity = 4  # количество пикселей для усреднения вокруг стартового пикслея
    _flood_fill_flags = (connectivity | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | 255 << 8)

    def __init__(self, path):
        """
        Основное окно для Медицинского разметчика.
        :param path: DICOM путь для чтения и разметки
        """
        # Читаем DICOM
        self.survey, self.image, self.base_windowing = self.read_survey(path)
        self.image_shape = self.image.shape[:2]  # размеры изображения
        self.mask = np.zeros(self.image_shape, dtype=np.uint8)  # маска с разметкой
        self.flood_mask = np.zeros(self.image_shape, dtype=np.uint8)  # маска с разметкой объекта заливкой

        # Динамические значения, которые будет изменять врач во время разметки
        self.windowing = self.base_windowing.copy()  # Значения для windowing'а
        self.tolerance = (10,) * 3  # максимальное отклонение значения пикселя при разметке заливкой
        self.brush_size = 5  # толщина кисти

        # Флаги
        self._draw_flag = False  # рисуем ли
        self._erase_flag = False  # стираем ли
        self._floodfill_flag = False  # активирована ли маска заливки

        # Инициализация окна и виджетов
        self._init_window()
        self._init_widgets()

    @staticmethod
    def read_survey(path):
        """Чтение DICOM исследования. Возвращает объект DICOM, исследование и значения windowing'а"""
        survey = pydicom.dcmread(path)
        img, base_windowing = dt.dicom2image(survey, equalize=False, raw=True)
        img = dt.window_image(img, *base_windowing)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return survey, img, base_windowing

    def _init_window(self):
        cv2.namedWindow(self.name)

    def _init_widgets(self):
        """Инициализирует виджеты"""
        # Слайдеры
        cv2.createTrackbar("Tolerance", self.name, self.tolerance[0], 127, self._tolerance_callback)
        cv2.createTrackbar("WC", self.name, self.windowing[0], 2048, self._wc_callback)
        cv2.createTrackbar("WW", self.name, self.windowing[1], 4096, self._ww_callback)
        cv2.createTrackbar("brush size", self.name, self.brush_size, 20, self._brush_size_callback)

        # Обработка мыши
        cv2.setMouseCallback(self.name, self._mouse_callback)

    def _update_windowing(self):
        self.image, _ = dt.dicom2image(self.survey, equalize=False, raw=True)
        self.image = dt.window_image(self.image, *self.windowing)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

    def _floodfill(self):
        """Разметка заливкой."""
        self.flood_mask = np.zeros(self.image_shape, dtype=np.uint8)
        all_contours = find_exterior_contours(self.positive_mask)
        image = np.array(self.image.copy() // 2, dtype=np.uint8)
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
        self._update_image()

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

    def _update_image(self):
        """Обновляет изображение и заново его отрисовывает."""
        mask = self.positive_mask
        if self._floodfill_flag:
            mask = cv2.bitwise_or(mask, np.array(self.flood_mask, dtype=np.uint8))
        image = self._apply_mask(self.image, mask, color=COLOR_GREEN)
        image = self._apply_mask(image, self.negative_mask, color=COLOR_RED)
        cv2.imshow(self.name, image)

    def _draw_circle(self, x, y):
        if self._draw_flag:
            cv2.circle(self.mask, (x, y), self.brush_size, COLOR_POSITIVE, -1)
        elif self._erase_flag:
            cv2.circle(self.mask, (x, y), self.brush_size, COLOR_NEGATIVE, -1)
        self._update_image()

    def _wc_callback(self, pos):
        self.windowing[0] = pos
        self._update_windowing()
        self._update_image()

    def _ww_callback(self, pos):
        self.windowing[1] = pos
        self._update_windowing()
        self._update_image()

    def _brush_size_callback(self, pos):
        self.brush_size = pos

    def _tolerance_callback(self, pos):
        self.tolerance = (pos,) * 3

    def _get_tolerance(self):
        return cv2.getTrackbarPos("Tolerance", self.name)

    def _set_tolerance(self, value):
        cv2.setTrackbarPos("Tolerance", self.name, min(max(0, value), 255))

    def _mouse_callback(self, event, x, y, flags, *userdata):
        """Обработка событий мыши."""

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

        # Изменение коэффициента tolerance
        if event == cv2.EVENT_MOUSEWHEEL:  # Вращение колёсика мыши
            # Если колёсико мыши - вверх, то увеличиваем значение, иначе - уменьшаем
            tolerance_increment_flag = 1 if flags > 0 else -1
            self._set_tolerance(self._get_tolerance() + tolerance_increment_flag)

            # Разметка заливкой
            self._floodfill()

    def _keyboard_callback(self, key):
        """Обработка событий клавиатуры."""
        if key in (ord("q"), ord("\x1b")):
            cv2.destroyWindow(self.name)
            return APP_FLAG_CLOSE_WINDOW
        elif key == ord("f"):
            self._floodfill_flag = not self._floodfill_flag
            print("Floodfill " + "activated" if self._floodfill_flag else "disabled")
            self._floodfill()

    @property
    def positive_mask(self):
        return np.array(self.mask == 1, dtype=np.uint8)

    @property
    def negative_mask(self):
        return np.array(self.mask == 2, dtype=np.uint8)

    def show(self):
        """Отображает окно."""
        self._update_image()
        print("Press [Q] or [ESC] to close the window.")
        print("Press [F] to activate/disable floodfill mask")
        while True:
            key = cv2.waitKey() & 0xFF
            traceback = self._keyboard_callback(key)
            if traceback == APP_FLAG_CLOSE_WINDOW:
                return
