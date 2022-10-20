import pydicom
import cv2

import numpy as np

import utils.preprocessing.dicom_transforms as upd
import utils.preprocessing.marking as upm

dicom_path = 'test_data/liver_001.dcm'
survey = pydicom.dcmread(dicom_path)
image = upd.dicom2image(survey, equalize=False)
denoised = upm.denoise(image, power=13)
thresholded = upm.threshold(denoised)
contour_image = upm.transform_to_contours(thresholded)
filtered = upm.remove_small_dots(contour_image)

denoised_color = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
denoised_color[filtered == 255] = (0, 0, 128)

cv2.imshow('Original', image)
cv2.imshow('Denoised', denoised)
cv2.imshow('After transforms', denoised_color)
cv2.waitKey(0)
