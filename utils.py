__author__ = 'mohamed'
import numpy as np

import cv2

import rects

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

        cv2.imshow('frame', frame)


def face_regions():
    x, y, w, h = (0, 0, 200, 200)
    white = (255, 255, 255)
    img = read_gray('momo.pgm')
    searchRect = (x + w / 7, y, w * 2 / 7, h / 2)
    rects.outline_rect(img, searchRect, white)
    searchRect = (x + w * 4 / 7, y, w * 2 / 7, h / 2)
    rects.outline_rect(img, searchRect, white)
    searchRect = (x + w / 4, y + h / 4, w / 2, h / 2)
    rects.outline_rect(img, searchRect, white)
    searchRect = (x + w / 6, y + h * 2 / 3, w * 2 / 3, h / 3)
    rects.outline_rect(img, searchRect, white)

    cv2.imwrite('momo_outlined.png', img)

def main():
    face_regions()


if __name__ == '__main__':
    main()

