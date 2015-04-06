__author__ = 'mohamed'

import Image
import os
import csv
import random
import math
import numpy as np

import cv
import cv2



# Reads image before training
def read_matrix_file(filename):
    return cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)


# Reads the csv
def read_csv(filename='names.csv'):
    csv_file = open(filename, 'r')
    return csv_file


# Compute the distance between two points in an image
def distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


# Prepares the  training CSV from the file system
def init_csv():
    BASE_PATH = "pictures"
    SEPARATOR = ";"

    ifile = open("/home/mohamed/PycharmProjects/FaceRecoTP/test_names.csv", "wb")
    writer = csv.writer(ifile, delimiter=';')
    data = []
    label = 0
    corresponding = []

    for dirname, dirnames, filenames in os.walk(BASE_PATH):
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

    print corresponding
    ifile.close()


def split_test_training_data(data, ratio=0.2):
    test_size = int(math.floor(ratio * len(data)))
    random.shuffle(data)
    return data[test_size:], data[:test_size]


def prepare_training(file):
    lines = file.readlines()
    training_data, testing_data = split_test_training_data(lines)
    return training_data, testing_data


def normalize_face_size(face):
    normalized_face_dimensions = (200, 200)
    resized_face = cv2.resize(np.asarray(face), normalized_face_dimensions)
    resized_face = cv.fromarray(resized_face)
    return resized_face


# Transforms an image: resizing, rotating and translating
def scale_rotate_translate(image, angle, center=None, new_center=None, scale=None, resample=Image.BICUBIC):
    if (scale is None) and (center is None):
        return image.rotate(angle=angle, resample=resample)

    nx, ny = x, y = center
    sx = sy = 1.0

    if new_center:
        (nx, ny) = new_center
    if scale:
        (sx, sy) = (scale, scale)

    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine / sx
    b = sine / sx
    c = x - nx * a - ny * b
    d = -sine / sy
    e = cosine / sy
    f = y - nx * d - ny * e

    return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=resample)


# Returns our area of interest: a cropped face
def crop_face(image, eye_left=(0, 0), eye_right=(0, 0), offset_pct=(0.2, 0.2), destination_size=(70, 70)):
    # calculate offsets in original image
    offset_h = math.floor(float(offset_pct[0]) * destination_size[0])
    offset_v = math.floor(float(offset_pct[1]) * destination_size[1])
    # get the direction
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    # calc rotation angle in radians
    rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
    # distance between them
    dist = distance(eye_left, eye_right)
    # compute the reference eye-width
    reference = destination_size[0] - 2.0 * offset_h
    # scale factor
    scale = float(dist) / float(reference)
    # rotate original around the left eye
    image = scale_rotate_translate(image, center=eye_left, angle=rotation)
    # crop the rotated image
    crop_xy = (eye_left[0] - scale * offset_h, eye_left[1] - scale * offset_v)
    crop_size = (destination_size[0] * scale, destination_size[1] * scale)
    image = image.crop(
        (int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0] + crop_size[0]), int(crop_xy[1] + crop_size[1])))
    # resize it
    image = image.resize(destination_size, Image.ANTIALIAS)
    return image


# Get the job done
def main():
    training_data, testing_data = prepare_training(read_csv("test_names.csv"))


if __name__ == '__main__':
    main()