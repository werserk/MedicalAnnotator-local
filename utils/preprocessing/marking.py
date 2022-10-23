from cv2 import cv2
import numpy as np


def denoise(image, power=13, temp_window_size=7, search_window_size=21):
    denoised = cv2.fastNlMeansDenoising(image, None, power, temp_window_size, search_window_size)
    return denoised


def threshold(image):
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 1)
    return thresh


def transform_to_contours(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
    return image


def remove_small_dots(image):
    binary_map = image.copy()
    binary_map = 255 - binary_map  # invert
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)
    areas = stats[1:, cv2.CC_STAT_AREA]
    result = np.zeros(labels.shape, np.uint8)
    for i in range(nlabels - 1):
        if areas[i] <= 10:
            result[labels == i + 1] = 1
    result = cv2.bitwise_or(result, image)
    return result


def erode(image, kernel_size):
    kernel = np.ones(kernel_size, np.uint8)
    eroded = cv2.erode(image, kernel)
    return eroded
