from cv2 import cv2


def find_contours(img, flag=cv2.RETR_TREE):
    # Находим контуры объектов разметки
    ret = cv2.findContours(img, flag, cv2.CHAIN_APPROX_SIMPLE)
    if len(ret) == 2:
        return ret[0]
    if len(ret) == 3:
        return ret[1]
