import cv2
import numpy as np
import pydicom
import utils.preprocessing.dicom_transforms as dt

SHIFT_KEY = cv2.EVENT_FLAG_SHIFTKEY
ALT_KEY = cv2.EVENT_FLAG_ALTKEY


def _find_exterior_contours(img):
    ret = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(ret) == 2:
        return ret[0]
    elif len(ret) == 3:
        return ret[1]
    raise Exception("Check the signature for `cv.findContours()`.")


class SelectionWindow:
    def __init__(self, path, name="Magic Wand Selector", connectivity=4, tolerance=20):
        self.survey, self.img, self.base_windowing = self.read_img(path)
        self.name = name
        self.img_shape = h, w = self.img.shape[:2]
        self.mask = np.zeros((h, w), dtype=np.uint8)
        self._flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        self._flood_fill_flags = (
                connectivity | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | 255 << 8
        )  # 255 << 8 tells to fill with the value 255

        # adaptive vars
        self.tolerance = (tolerance,) * 3
        self.windowing = self.base_windowing.copy()
        self.coords = []
        self.brush_size = 3

        # widgets
        cv2.namedWindow(self.name)
        self._init_widgets()

    @staticmethod
    def read_img(path):
        survey = pydicom.dcmread(path)
        img, base_windowing = dt.dicom2image(survey, equalize=False, raw=True)
        img = dt.window_image(img, *base_windowing)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return survey, img, base_windowing

    def _init_widgets(self):
        cv2.createTrackbar("Tolerance", self.name, self.tolerance[0], 255, self._tolerance_callback)
        cv2.createTrackbar('WC', self.name, self.windowing[0], 2048, self._wc_callback)
        cv2.createTrackbar('WW', self.name, self.windowing[1], 4096, self._ww_callback)
        cv2.createTrackbar('brush size', self.name, self.brush_size, 20, self._brush_size_callback)
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
        if len(self.coords) == 0:
            return
        self.mask = np.zeros(self.img_shape, dtype=np.uint8)

        for x, y in self.coords:
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

        # if modifier == (ALT_KEY + SHIFT_KEY):
        #     self.mask = cv.bitwise_and(self.mask, flood_mask)
        # elif modifier == ALT_KEY:
        #     self.mask = cv.bitwise_and(self.mask, cv.bitwise_not(flood_mask))
        # else:
        #     self.mask = flood_mask

    def _mouse_callback(self, event, x, y, flags, *userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.last_x, self.last_y = x, y
            modifier = flags & (ALT_KEY + SHIFT_KEY)
            if modifier == SHIFT_KEY:
                self.coords.append((x, y))
            else:
                self.coords = [(x, y)]
            self._floodfill(flags)
        if event == 10:
            if flags > 0:
                cv2.setTrackbarPos("Tolerance", self.name,
                                   min(255, cv2.getTrackbarPos("Tolerance", self.name) + 1))
            else:
                cv2.setTrackbarPos("Tolerance", self.name,
                                   max(0, cv2.getTrackbarPos("Tolerance", self.name) - 1))
            self._floodfill(flags)
        self._update()

    def _update(self):
        """Updates an image in the already drawn window."""
        viz = self.img.copy()
        contours = _find_exterior_contours(self.mask)
        viz = cv2.drawContours(viz, contours, -1, color=(255,) * 3, thickness=-1)
        viz = cv2.addWeighted(self.img, 0.75, viz, 0.25, 0)
        viz = cv2.drawContours(viz, contours, -1, color=(255,) * 3, thickness=1)

        cv2.imshow(self.name, viz)

    def show(self):
        """Draws a window with the supplied image."""
        self._update()
        print("Press [q] or [esc] to close the window.")
        while True:
            k = cv2.waitKey() & 0xFF
            if k in (ord("q"), ord("\x1b")):
                cv2.destroyWindow(self.name)
                break
