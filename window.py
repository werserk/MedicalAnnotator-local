import cv2
import numpy as np
import pydicom
import utils.preprocessing.dicom_transforms as dt

SHIFT_KEY = cv2.EVENT_FLAG_SHIFTKEY
ALT_KEY = cv2.EVENT_FLAG_ALTKEY


def _find_exterior_contours(img):
    # Находим контуры объектов разметки
    ret = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(ret) == 2:
        return ret[0]
    if len(ret) == 3:
        return ret[1]
    raise Exception("Check the signature for `cv.findContours()`.")


class SelectionWindow:
    name = 'Медицинский разметчик'
    connectivity = 4

    def __init__(self, path):
        """
        Основное окно для Медицинского разметчика
        :param path: DICOM путь для чтения и разметки
        """
        # Читаем DICOM
        self.survey, self.img, self.base_windowing = self.read_img(path)
        self.img_shape = self.img.shape[:2]  # размеры изображения
        self.mask = np.zeros(self.img_shape, dtype=np.uint8)  # пустая маска
        self._flood_mask = np.zeros((self.img_shape[0] + 2, self.img_shape[1] + 2),
                                    dtype=np.uint8)  # временная маска для разметки заливкой
        self._flood_fill_flags = (self.connectivity | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | 255 << 8)

        # Динамические значения, которые будет изменять врач во время разметки
        self.windowing = self.base_windowing.copy()  # Значения для windowing'а
        self.coordinates = []  # координаты покрашенных пикселей
        self.tolerance = (20,) * 3  # максимальное отклонение значения пикселя при разметке заливкой
        self.brush_size = 3  # толщина кисти

        # Инициализация окна и виджетов
        self._init_window()
        self._init_widgets()

    @staticmethod
    def read_img(path):
        """Чтение DICOM исследования. Возвращает объект DICOM, исследование и значения windowing'а"""
        survey = pydicom.dcmread(path)
        img, base_windowing = dt.dicom2image(survey, equalize=False, raw=True)
        img = dt.window_image(img, *base_windowing)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return survey, img, base_windowing

    def _init_window(self):
        cv2.namedWindow(self.name)

    def _init_widgets(self):
        """Инициализирует"""
        # Слайдеры
        cv2.createTrackbar("Tolerance", self.name, self.tolerance[0], 255, self._tolerance_callback)
        cv2.createTrackbar("WC", self.name, self.windowing[0], 2048, self._wc_callback)
        cv2.createTrackbar("WW", self.name, self.windowing[1], 4096, self._ww_callback)
        cv2.createTrackbar("brush size", self.name, self.brush_size, 20, self._brush_size_callback)

        # Обработка мыши
        cv2.setMouseCallback(self.name, self._mouse_callback)

    def _update_windowing(self):
        self.img, _ = dt.dicom2image(self.survey, equalize=False, raw=True)
        self.img = dt.window_image(self.img, *self.windowing)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)

    def _wc_callback(self, pos):
        self.windowing[0] = pos
        self._update_windowing()
        self._update()

    def _ww_callback(self, pos):
        self.windowing[1] = pos
        self._update_windowing()
        self._update()

    def _brush_size_callback(self, pos):
        self.brush_size = pos

    def _tolerance_callback(self, pos):
        self.tolerance = (pos,) * 3

    def _floodfill(self, flags):
        """Разметка заливкой."""
        if len(self.coordinates) == 0:
            return
        self.mask = np.zeros(self.img_shape, dtype=np.uint8)

        for x, y in self.coordinates:
            self._flood_mask[:] = 0
            cv2.floodFill(
                self.img,
                self._flood_mask,
                (x, y),
                0,
                self.tolerance,
                self.tolerance,
                self._flood_fill_flags,
            )
            flood_mask = self._flood_mask[1:-1, 1:-1].copy()
            self.mask = cv2.bitwise_or(self.mask, flood_mask)

    def _get_tolerance(self):
        return cv2.getTrackbarPos("Tolerance", self.name)

    def _set_tolerance(self, value):
        cv2.setTrackbarPos("Tolerance", self.name, min(max(0, value), 255))

    def _mouse_callback(self, event, x, y, flags, *userdata):
        """Обработка мыши."""
        # Разметка заливкой с отклонением не более tolerance
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.last_x, self.last_y = x, y
            modifier = flags & (ALT_KEY + SHIFT_KEY)
            if modifier == SHIFT_KEY:
                self.coordinates.append((x, y))
            else:
                self.coordinates = [(x, y)]
            self._floodfill(flags)

        # Изменение коэффициента tolerance
        if event == cv2.EVENT_MOUSEWHEEL:
            # Если колёсико мыши - вверх, то увеличиваем, иначе - уменьшаем
            tolerance_increment_flag = 1 if flags > 0 else -1
            self._set_tolerance(self._get_tolerance() + tolerance_increment_flag)

            # Разметка заливкой
            self._floodfill(flags)

        # Обновляем разметку
        self._update()

    def _update(self):
        """Обновляет изображение и заново его отрисовывает."""
        viz = self.img.copy()
        contours = _find_exterior_contours(self.mask)
        viz = cv2.drawContours(viz, contours, -1, color=(255,) * 3, thickness=-1)
        viz = cv2.addWeighted(self.img, 0.75, viz, 0.25, 0)
        viz = cv2.drawContours(viz, contours, -1, color=(255,) * 3, thickness=1)

        cv2.imshow(self.name, viz)

    def show(self):
        """Отображает окно."""
        self._update()
        print("Press [q] or [esc] to close the window.")
        while True:
            k = cv2.waitKey() & 0xFF
            if k in (ord("q"), ord("\x1b")):
                cv2.destroyWindow(self.name)
                break
