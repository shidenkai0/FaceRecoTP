__author__ = 'mohamed'
import numpy as np

import cv2

import trackers


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
    return cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)


def extract_faces_from_video(filename, output_dir=''):
    tracker = trackers.FaceTracker(scaleFactor=1.2, minNeighbors=2, flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
    video_cap = cv2.VideoCapture(filename)

    while video_cap.isOpened():
        ret, frame = video_cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame', frame)


def main():
    extract_faces_from_video('Alex.mpg')


if __name__ == '__main__':
    main()

