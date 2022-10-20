import pydicom
import cv2

import numpy as np

import utils.preprocessing.dicom_transforms as dt
from utils.preprocessing import marking

dicom_path = 'test_data/liver_001.dcm'
windowName = dicom_path.split('/')[-1]

survey = pydicom.dcmread(dicom_path)
original_image, base_windowing = dt.dicom2image(survey, equalize=False, raw=True)
windowing = base_windowing.copy()
image = original_image.copy()
H, W = image.shape
mask = np.zeros_like(image, dtype=np.uint8)
mouseX, mouseY = None, None
pixels = {}
brush_size = 3
drawing = False


def update_window():
    image = dt.window_image(original_image, *windowing)
    masked_image = apply_mask(image, mask)
    cv2.imshow(windowName, masked_image)


def ww_change(val):
    global image
    windowing[1] = val
    update_window()


def wc_change(val):
    global image
    windowing[0] = val
    update_window()


def mouse_event(event, x, y, flags, param):
    global mouseX, mouseY, drawing
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX, mouseY = x, y
        update_percents()
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        mouseX, mouseY = x, y
        paint()
        update_window()
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            mouseX, mouseY = x, y
            paint()
            update_window()
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    if event == 10:
        if flags > 0:
            cv2.setTrackbarPos('percents', windowName,
                               min(100, cv2.getTrackbarPos('percents', windowName) + 1))
        else:
            cv2.setTrackbarPos('percents', windowName,
                               max(0, cv2.getTrackbarPos('percents', windowName) - 1))


def paint():
    global mask, mouseY, mouseX
    mask[max(mouseY - brush_size, 0):min(mouseY + brush_size, H),
    max(mouseX - brush_size, 0): min(mouseX + brush_size, W)] = 1


def autolabel(image, x, y, percents):
    global mask, pixels
    if x is None:
        return mask
    mean_pixel_value = image[max(y - brush_size, 0):min(y + brush_size, H),
                       max(x - brush_size, 0): min(x + brush_size, W)].mean()
    pixels.add((mean_pixel_value, x, y))
    mask = recursive_paint(image, mask, x, y)
    return mask


def recursive_paint(image, mask, x, y):
    pass


def update_percents(val=None):
    global image, mask
    mask = autolabel(image, mouseX, mouseY, val)
    image = apply_mask(image, mask)
    cv2.imshow(windowName, image)


def update_brush(val):
    global brush_size
    brush_size = val


image = dt.window_image(original_image, *windowing)
image = apply_mask(image, mask)
cv2.imshow(windowName, image)

# Windowing trackbars


# On click
cv2.setMouseCallback(windowName, mouse_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
