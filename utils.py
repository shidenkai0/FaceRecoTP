__author__ = 'mohamed'
import numpy as np

import cv2


def is_gray(image):
    return image.ndim < 3


# returns the width and height of an image, divided by a scalar
def wh_divided_by(image, divisor):
    h, w = image.shape[:2]
    return w / divisor, h / divisor


def as_row_matrix(X):
    if len(X) == 0:
        return np.array([])
    return np.empty((0, X[0].size), dtype=X[0].dtype)


def read_gray(filename):
    print filename
    return cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)