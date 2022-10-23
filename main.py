from app.window import *
import cv2

if __name__ == "__main__":
    dicom_path = 'test_data/liver_001.dcm'

    window = SelectionWindow(dicom_path)

    window.show()
    cv2.destroyAllWindows()
