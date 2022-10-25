from cv2 import cv2
import numpy as np
import pydicom as dicom


def apply_windowing(img, window_center, window_width, intercept, slope, inverted=False):
    img = (img * slope + intercept)  # for translation adjustments given in the dicom file.
    img_min = window_center - window_width // 2  # minimum HU level
    img_max = window_center + window_width // 2  # maximum HU level
    img[img < img_min] = img_min  # set img_min for all HU levels less than minimum HU level
    img[img > img_max] = img_max  # set img_max for all HU levels higher than maximum HU level
    if inverted:
        img = -img
        img_min = img.min()
        img_max = img.max()
    img = (img - img_min) / (img_max - img_min) * 255.0
    img = np.array(img, dtype=np.uint8)
    return img


def get_first_of_dicom_field_as_int(x):
    # get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if isinstance(x, dicom.multival.MultiValue):
        x = x[0]
    x = str(x)
    x = x.split(',')[0]
    return int(float(x))


def get_windowing(data):
    # window center and width
    try:
        dicom_fields = [data[('0028', '1050')].value,  # window center
                        data[('0028', '1051')].value]  # window width
        res = [get_first_of_dicom_field_as_int(x) for x in dicom_fields]
    except KeyError:
        print('No window center or width')
        res = [2048, 4096]

    # intercept and slope
    try:
        dicom_fields = [data[('0028', '1052')].value,  # intercept
                        data[('0028', '1053')].value]  # slope
        res.extend([get_first_of_dicom_field_as_int(x) for x in dicom_fields])
    except KeyError:
        res.extend([0, 1])

    # is inverted
    try:
        res.append(data[('0028', '0004')].value == 'MONOCHROME1')
    except KeyError:
        res.append(False)
    return res


def equalize_hist(image):
    image = np.array(image, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    return image


def dicom2image(scan, raw=False, equalize=False):
    windowing = get_windowing(scan)
    image = scan.pixel_array
    if not raw:
        image = apply_windowing(image, *windowing)
    if equalize:
        image = equalize_hist(image)
    if raw:
        return image, windowing
    return image
