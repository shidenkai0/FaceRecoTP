__author__ = 'mohamed'
import math

import cv2

import utils
import rects


class Face(object):
    """ Object Representation of a face as several rectangles
    """

    def __init__(self):
        self.faceRect = None
        self.leftEyeRect = None
        self.rightEyeRect = None
        self.noseRect = None
        self.mouthRect = None

    def normalized_face(self, image):
        normalized_face_dimensions = (200, 200)
        x, y, w, h = self.faceRect
        resized_face = cv2.resize(image[y:y + h, x:x + w], normalized_face_dimensions, interpolation=cv2.INTER_LINEAR)
        return resized_face

    def show_face(self, image):
        x, y, w, h = self.faceRect
        cv2.imshow('face', image[y:y + h, x:x + w])

    def proper_face(self):
        if self.rightEyeRect is None or self.leftEyeRect is None:
            return None
        x_re, y_re, w_re, h_re = self.rightEyeRect
        x_le, y_le, w_le, h_le = self.leftEyeRect
        le_to_re_vector = (float((x_re + w_re / 2) - (x_le + w_le / 2)), float((y_re + h_re / 2) - (y_le + h_le / 2)))
        return math.atan2(le_to_re_vector[1], le_to_re_vector[0])


class FaceTracker(object):
    """ Tracks Facial Features in a frame """

    def __init__(self, scaleFactor=1.2,
                 minNeighbors=2,
                 flags=cv2.cv.CV_HAAR_SCALE_IMAGE):

        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.flags = flags
        self._faces = []

        self._faceClassifier = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml')
        self._eyeClassifier = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_eye.xml')
        self._noseClassifier = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_mcs_nose.xml')
        self._mouthClassifier = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_mcs_mouth.xml')

    @property
    def faces(self):
        """
        Returns the list of detected faces
        """
        return self._faces

    def _detect_one_object(self, classifier, image, rect, imageSizeToMinSize):
        x, y, w, h = rect

        min_size = utils.wh_divided_by(image, imageSizeToMinSize)
        region_of_interest = image[y:y + h, x:x + w]
        detected_rects = classifier.detectMultiScale(region_of_interest, self.scaleFactor, self.minNeighbors,
                                                     self.flags, min_size)

        if len(detected_rects) == 0:
            return None

        x_roi, y_roi, w_roi, h_roi = detected_rects[0]
        return x + x_roi, y + y_roi, w_roi, h_roi

    def refresh(self, image):

        self._faces = []

        if utils.is_gray(image):
            image = cv2.equalizeHist(image)
        else:
            image = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
            cv2.equalizeHist(image, image)

        minSize = utils.wh_divided_by(image, 8)

        faceRects = self._faceClassifier.detectMultiScale(image, self.scaleFactor,
                                                          self.minNeighbors, self.flags, minSize)

        if faceRects is not None:
            for faceRect in faceRects:
                face = Face()
                face.faceRect = faceRect
                x, y, w, h = faceRect
                # Detect left eye
                searchRect = (x + w / 7, y, w * 2 / 7, h / 2)
                face.leftEyeRect = self._detect_one_object(self._eyeClassifier, image, searchRect, 64)
                # Detect right eye
                searchRect = (x + w * 4 / 7, y, w * 2 / 7, h / 2)
                face.rightEyeRect = self._detect_one_object(self._eyeClassifier, image, searchRect, 64)
                # Detect Nose
                searchRect = (x + w / 4, y + h / 4, w / 2, h / 2)
                face.noseRect = self._detect_one_object(self._noseClassifier, image, searchRect, 32)
                # Detect mouth
                searchRect = (x + w / 6, y + h * 2 / 3, w * 2 / 3, h / 3)
                face.mouthRect = self._detect_one_object(self._mouthClassifier, image, searchRect, 16)

                self._faces.append(face)

    def drawDebugRects(self, image):
        """
        Draw rectangles around the detected objects
        :param image:
        :return:
        """

        if utils.is_gray(image):
            faceColor = 255
            leftEyeColor = 255
            righEyeColor = 255
            noseColor = 255
            mouthColor = 255

        else:
            faceColor = (255, 255, 0)
            leftEyeColor = (0, 255, 0)
            righEyeColor = (255, 0, 0)
            noseColor = (255, 0, 255)
            mouthColor = (0, 0, 255)

        for face in self._faces:
            rects.outline_rect(image, face.faceRect, faceColor)
            rects.outline_rect(image, face.leftEyeRect, leftEyeColor)
            rects.outline_rect(image, face.rightEyeRect, righEyeColor)
            rects.outline_rect(image, face.noseRect, noseColor)
            rects.outline_rect(image, face.mouthRect, mouthColor)
