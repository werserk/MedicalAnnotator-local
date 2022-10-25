from cv2 import cv2
import pydicom
from utils.preprocessing.dicom_transforms import dicom2image, apply_windowing
from app.constants import *


class BaseWindow:
    name = 'Medical annotator'

    def __init__(self, path):
        """
        Основное окно для Медицинского разметчика.
        :param path: DICOM путь для чтения и разметки
        """
        # Читаем DICOM
        self.survey, self.image, self.base_windowing = self.read_survey(path)
        self.image_shape = self.image.shape[:2]  # размеры изображения

        # Динамические значения, которые будет изменять врач во время разметки
        self.windowing = self.base_windowing.copy()  # Значения для windowing'а
        if type(self) is BaseWindow:
            self._init_window()
            self._init_widgets()

    @staticmethod
    def read_survey(path):
        """Чтение DICOM исследования. Возвращает объект DICOM, исследование и значения windowing'а"""
        survey = pydicom.dcmread(path)
        img, base_windowing = dicom2image(survey, equalize=False, raw=True)
        img = apply_windowing(img, *base_windowing)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return survey, img, base_windowing

    def _init_widgets(self):
        """Инициализирует виджеты."""
        cv2.createTrackbar("WC", self.name, self.windowing[0], 2048, self._wc_callback)
        cv2.createTrackbar("WW", self.name, self.windowing[1], 4096, self._ww_callback)
        cv2.setMouseCallback(self.name, self._mouse_callback)

    def _wc_callback(self, pos):
        self.windowing[0] = pos
        self._update_windowing()
        self._update_image()

    def _ww_callback(self, pos):
        self.windowing[1] = pos
        self._update_windowing()
        self._update_image()

    def _update_windowing(self):
        self.image, _ = dicom2image(self.survey, equalize=False, raw=True)
        self.image = apply_windowing(self.image, *self.windowing)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

    def _update_image(self):
        """Обновляет изображение и заново его отрисовывает."""
        cv2.imshow(self.name, self.image)

    def _init_window(self):
        cv2.namedWindow(self.name)

    def _mouse_callback(self, event, x, y, flags, *userdata):
        pass

    def _keyboard_callback(self, key):
        """Обработка событий клавиатуры."""
        if key in (ord("q"), ord("\x1b")):
            cv2.destroyWindow(self.name)
            return APP_FLAG_CLOSE_WINDOW

    def show(self):
        """Отображает окно."""
        self._update_image()
        print(f"{self.__class__.__name__} is currently working")
        print("Press [Q] or [ESC] to close the window.")
        while True:
            key = cv2.waitKey() & 0xFF
            traceback = self._keyboard_callback(key)
            if traceback == APP_FLAG_CLOSE_WINDOW:
                return
