__author__ = 'mohamed'

import sys
import datetime
import os

import cv2

import trackers
import recognizers


def main():
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    cam = cv2.VideoCapture(video_src)
    cam.set(3, 1280)
    cam.set(4, 720)

    tracker = trackers.FaceTracker(scaleFactor=1.2, minNeighbors=2, flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
    eigen_recognizer = recognizers.EigenRecognizer()
    fisher_recognizer = recognizers.FisherRecognizer()

    # Main loop
    while True:
        ret, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.cv.CV_BGR2GRAY)
        # frame = cv2.equalizeHist(frame)
        tracker.refresh(frame)
        dirname = 'pictures'

        for face in tracker.faces:
            if face.proper_face():
                norm_face = face.normalized_face(frame)
                cv2.imshow('Norm Face', norm_face)

                print("Eigen Result: ")
                print eigen_recognizer.predict(norm_face)
                print "Fisher Result :"
                print fisher_recognizer.predict(norm_face)

        cv2.imshow('momo', frame)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
        elif ch == 114:
            temps = datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')

            for face in tracker.faces:
                cv2.imwrite(os.path.join(dirname, 'saving' + temps + '.pgm'), face.normalized_face(frame))


if __name__ == '__main__':
    main()