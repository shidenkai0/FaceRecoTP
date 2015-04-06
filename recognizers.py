__author__ = 'mohamed'

import csv
import os
import numpy as np

import cv2

import utils


class RecognizerTest(object):
    def __init__(self, recognizer, db_path="pictures", test_path="test"):
        assert isinstance(recognizer, FaceRecognizer)
        self.Recognizer = recognizer
        self.TEST_PATH = test_path
        self.DB_PATH = self.Recognizer.db_manager.IMAGES_PATH

    def testReco(self):
        print 1


class FaceDbManager(object):
    def __init__(self, images_path='pictures', csv_path='names.csv'):
        self.IMAGES_PATH = images_path
        self._faces_dictionary = None
        self.CSV_PATH = csv_path
        self.LabelNameTable = None

    def create_dict(self):
        label_dict = {}
        csv_file = open(self.CSV_PATH, mode='r')

        for line in csv_file:

            filename, label = line.strip().split(';')
            # update the current key if it exists, else append to it
            if label_dict.has_key(int(label)):
                current_files = label_dict.get(label)
                np.append(current_files, utils.read_gray(filename))
            else:
                label_dict[int(label)] = utils.read_gray(filename)

            self._faces_dictionary = label_dict

    @property
    def faces_dictionary(self):
        return self._faces_dictionary

    def create_csv(self):
        SEPARATOR = ";"

        csv_file = open(self.CSV_PATH, "wb")
        writer = csv.writer(csv_file, delimiter=';')
        data = []
        label = 0
        corresponding = []

        for dirname, dirnames, filenames in os.walk(self.IMAGES_PATH):
            for subdirname in dirnames:
                subject_path = os.path.join(dirname, subdirname)
                for filename in os.listdir(subject_path):
                    abs_path = "%s/%s" % (subject_path, filename)
                    row = "%s%s%s" % (abs_path, SEPARATOR, label)
                    data.append(row.split(";"))
                corresponding.append(subdirname)
                label += 1

        for line4 in data:
            writer.writerow(line4)

        self.LabelNameTable = corresponding
        csv_file.close()


class FaceRecognizer(object):
    def __init__(self):
        self.db_manager = FaceDbManager()
        self.db_manager.create_csv()
        self.db_manager.create_dict()
        self.recognizer = None

    def predict(self, face_image):
        return self.recognizer.predict(face_image)


class EigenRecognizer(FaceRecognizer):
    def __init__(self):
        super(EigenRecognizer, self).__init__()
        self.recognizer = cv2.createEigenFaceRecognizer()
        self.recognizer.train(self.db_manager.faces_dictionary.values(),
                              np.array(self.db_manager.faces_dictionary.keys()))


class FisherRecognizer(FaceRecognizer):
    def __init__(self):
        super(FisherRecognizer, self).__init__()
        self.recognizer = cv2.createFisherFaceRecognizer()
        self.recognizer.train(self.db_manager.faces_dictionary.values(),
                              np.array(self.db_manager.faces_dictionary.keys()))



