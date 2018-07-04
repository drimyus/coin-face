import cv2
import numpy as np
import dlib
import os
import sys


# dlib's landmarks
NOSE_POINTS = [4]
RIGHT_EYE_POINTS = list(range(2, 4))
LEFT_EYE_POINTS = list(range(0, 2))
NUM_TOTAL_POINTS = 5

# location of detector model
cur = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(cur, os.pardir))

DETECTOR_DIR = os.path.join(ROOT, "model")
DETECTOR_MODE = "haar"  # "dlib"


JAW_POINTS = list(range(0, 17))
FACE_POINTS = list(range(17, 68))

# dlib landmarks indexing ...
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_POINTS = list(range(48, 68))


# adjust settings
LEFT_EYE_SCALE = 2.5
RIGHT_EYE_SCALE = 2.5
MOUSE_SCALE = 2.5

COLOR_MODE = "BGR"  # "GRAY"  # "CUSTOM"


