from window import *
import cv2
import pydicom
import utils.preprocessing.dicom_transforms as dt

if __name__ == "__main__":
    dicom_path = 'test_data/liver_001.dcm'

    window = SelectionWindow(dicom_path, "Magic Wand Selector")

    print("Click to seed a selection.")
    print(" * [SHIFT] adds to the selection.")
    print(" * [ALT] subtracts from the selection.")
    print(" * [SHIFT] + [ALT] intersects the selections.")
    print()

    window.show()
    cv2.destroyAllWindows()
