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
        self.tolerance = 10  # максимальное отклонение значения пикселя при разметке заливкой

        self._floodfill_flag = False  # активирована ли маска заливки

        # Инициализация окна и виджетов
        if type(self) is FloodFillWindow:
            self._init_window()
            self._init_widgets()

    def _init_widgets(self):
        """Инициализирует виджеты"""
        super(FloodFillWindow, self)._init_widgets()
        # Слайдеры
        cv2.createTrackbar("Tolerance", self.name, self.tolerance, 50, self._tolerance_callback)

        # Обработка мыши
        cv2.setMouseCallback(self.name, self._mouse_callback)

    def _single_floodfill(self, image, x, y, tolerance):
        # Создаём специальную временную пустую маску для заливки
        _flood_mask = np.zeros((self.image_shape[0] + 2, self.image_shape[1] + 2), dtype=np.uint8)

        # Процесс заливки
        cv2.floodFill(image, _flood_mask, (x, y), 0, tolerance, tolerance, self._flood_fill_flags)

        # Обрезаем специальную маску (такова специфика функции)
        flood_mask = _flood_mask[1:-1, 1:-1].copy()
        return flood_mask

    def _floodfill(self):
        """Разметка заливкой."""
        # Обнуляем маску с заливкой
        self.flood_mask = np.zeros(self.image_shape, dtype=np.uint8)

        # Получаем контуры всех объектов
        all_contours = find_exterior_contours(self.positive_mask)

        # Переводим в [0..1] из-за особенностей функции floodfill
        tolerance = (self.tolerance / 1000,) * 3
        image = np.array(self.blurred_image.copy() / 1000, dtype=np.float32)

        # Создаём запретную зону для floodfill (красный цвет при разметке)
        image[self.mask == 2] = 1.0

        for contours in all_contours:  # Итерируемся для каждого объекта
            for coordinates in contours:  # Итерируемся для каждой пары координат
                x, y = coordinates[0]  # Нас интересуют только пары координат
                flood_mask = self._single_floodfill(image, x, y, tolerance)  # Заливка
                self.flood_mask = cv2.bitwise_or(self.flood_mask, flood_mask)  # Объединяем с ранее созданными масками

            # Такую же процедуру сделаем для центра объекта
            centroid = contours.mean(axis=1).mean(axis=0)  # Находим примерный центр
            xc, yc = int(centroid[0]), int(centroid[1])  # Переводим в int
            if self.positive_mask[xc][yc] == 1:  # Если центр является размеченным (а то может быть кольцо)
                flood_mask = self._single_floodfill(image, xc, yc, tolerance)  # Заливка
                self.flood_mask = cv2.bitwise_or(self.flood_mask, flood_mask)  # Объединяем с ранее созданными масками
        self.flood_mask = mkg.remove_small_dots(self.flood_mask)  # Убираем мелкие точки
        self._update_image()

    def _update_image(self):
        """Обновляет изображение и заново его отрисовывает."""
        positive_mask = self.positive_mask
        if self._floodfill_flag:  # Если включен режим с заливкой, то объединяем
            positive_mask = cv2.bitwise_or(self.positive_mask, np.array(self.flood_mask, dtype=np.uint8))

        # Получаем маски и контуры
        positive_mask, positive_contours = self.apply_mask(positive_mask, color=COLOR_GREEN)
        negative_mask, negative_contours = self.apply_mask(self.negative_mask, color=COLOR_RED)
        _, cursor_contours = self.apply_mask(self.ui_mask, color=COLOR_WHITE)

        # Складываем маски и накладываем на изображение
        image = cv2.addWeighted(self.image, 0.75, positive_mask + negative_mask, 0.25, 0)

        # Рисуем поверх контуры
        image = cv2.drawContours(image, positive_contours, -1, color=COLOR_GREEN, thickness=1)
        image = cv2.drawContours(image, negative_contours, -1, color=COLOR_RED, thickness=1)
        image = cv2.drawContours(image, cursor_contours, -1, color=COLOR_WHITE, thickness=1)

        cv2.imshow(self.name, image)

    def _tolerance_callback(self, pos):
        self.tolerance = pos

    def _set_tolerance(self, value):
        cv2.setTrackbarPos("Tolerance", self.name, min(max(0, value), 255))

    def _mouse_callback(self, event, x, y, flags, *userdata):
        """Обработка событий мыши."""
        super(FloodFillWindow, self)._mouse_callback(event, x, y, flags, *userdata)
        # Изменение коэффициента tolerance
        if event == cv2.EVENT_MOUSEWHEEL:  # Вращение колёсика мыши
            # Если колёсико мыши - вверх, то увеличиваем значение, иначе - уменьшаем
            tolerance_increment_flag = 1 if flags > 0 else -1
            self._set_tolerance(self.tolerance + tolerance_increment_flag)

            # Разметка заливкой
            self._floodfill()

    def _keyboard_callback(self, key):
        """Обработка событий клавиатуры."""
        traceback = super(FloodFillWindow, self)._keyboard_callback(key)
        if traceback is not None:  # Если поступил какой-то сигнал, то сразу отправляем его
            return traceback
        if key == ord("f"):  # Если нажата F, то включаем/выключаем режим заливки
            self._floodfill_flag = not self._floodfill_flag
            print("Floodfill " + "activated" if self._floodfill_flag else "disabled")
            self._floodfill()
