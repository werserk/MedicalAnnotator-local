from cv2 import cv2


def find_exterior_contours(img):
    # Находим контуры объектов разметки
    ret = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(ret) == 2:
        return ret[0]
    if len(ret) == 3:
        return ret[1]
    raise Exception("Check the signature for `cv.findContours()`.")
