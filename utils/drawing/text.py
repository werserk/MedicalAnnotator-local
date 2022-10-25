from cv2 import cv2
from app.constants import *


def draw_text(image, text, pos,
              font=cv2.FONT_HERSHEY_PLAIN,
              font_scale=1,
              font_thickness=1,
              text_color=COLOR_BLACK,
              text_color_bg=COLOR_GREEN
              ):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    image = cv2.rectangle(image, pos, (x + text_w, y + text_h), text_color_bg, -1)
    image = cv2.putText(image, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    return image
