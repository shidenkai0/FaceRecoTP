__author__ = 'mohamed'

import Image
import sys
import os
import datetime
import csv
import random
import math
import numpy as np

import cv
import cv2


IMAGE_SCALE = 2
normalize_face_dimensions = (100, 100)

face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_eye.xml')


def create_dict(label_matrix):
    model = cv2.createEigenFaceRecognizer()
    model.train(label_matrix.values(), np.array(label_matrix.keys()))
    return model


def read_gray(filename):
    print filename
    return cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)


def predict_image_from_model(model, image):
    return model.predict(image)


def distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


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


def crop_face(image, eye_left=(0, 0), eye_right=(0, 0), offset_pct=(0.2, 0.2), dest_sz=(70, 70)):
    # calculate offsets in original image
    offset_h = math.floor(float(offset_pct[0]) * dest_sz[0])
    offset_v = math.floor(float(offset_pct[1]) * dest_sz[1])
    # get the direction
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    # calc rotation angle in radians
    rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
    # distance between them
    dist = distance(eye_left, eye_right)
    # calculate the reference eye-width
    reference = dest_sz[0] - 2.0 * offset_h
    # scale factor
    scale = float(dist) / float(reference)
    # rotate original around the left eye
    image = scale_rotate_translate(image, center=eye_left, angle=rotation)
    # crop the rotated image
    crop_xy = (eye_left[0] - scale * offset_h, eye_left[1] - scale * offset_v)
    crop_size = (dest_sz[0] * scale, dest_sz[1] * scale)
    image = image.crop(
        (int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0] + crop_size[0]), int(crop_xy[1] + crop_size[1])))
    # resize it
    image = image.resize(dest_sz, Image.ANTIALIAS)
    return image


def normalize_face_size(face):
    normalized_face_dimensions = (200, 200)
    resized_face = cv2.resize(np.asarray(face), normalized_face_dimensions)
    resized_face = cv.fromarray(resized_face)
    return resized_face


def read_csv(filename='names.csv'):
    csv = open(filename, 'r')
    return csv


def prepare_training(file):
    lines = file.readlines()
    training_data, testing_data = split_test_training_data(lines)
    return training_data, testing_data


def create_label_matrix_dict(input_file):
    """ Create dict of label -> matricies from file """
    ### for every line, if key exists, insert into dict, else append
    label_dict = {}

    for line in input_file:
        ## split on the ';' in the csv separating filename;label
        filename, label = line.strip().split(';')
        ##update the current key if it exists, else append to it
        if label_dict.has_key(int(label)):
            current_files = label_dict.get(label)
            np.append(current_files, read_gray(filename))
        else:
            label_dict[int(label)] = read_gray(filename)

    return label_dict


def split_test_training_data(data, ratio=0.2):
    test_size = int(math.floor(ratio * len(data)))
    random.shuffle(data)
    return data[test_size:], data[:test_size]


def as_row_matrix(X):
    if len(X) == 0:
        return np.array([])
    mat = np.empty((0, X[0].size), dtype=X[0].dtype)


def main():
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
    # myfile = open('names.csv','wb')
    #wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #wr.writerow(listefile)
    print corresponding
    ifile.close()
    csvfile = open("test_names.csv", 'r')

    training_data, testing_data = prepare_training(csvfile)

    data_dict = create_label_matrix_dict(training_data)
    print data_dict

    model = create_dict(data_dict)

    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    cam = cv2.VideoCapture(video_src)
    cam.set(3, 1280)
    cam.set(4, 720)
    print cam
    i = 0

    # Main loop
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            #for (ex,ey,ew,eh) in eyes:
            #   cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        cv2.imshow('Momo', frame)

        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
        elif ch == 115:
            temps = datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')
            crop_frame = frame[y:y + h, x:x + w]
            cv2.imshow("cropped", crop_frame)
            dirname = 'pictures'

            crop_frame2 = normalize_face_size(crop_frame)
            # cv2.imshow("cropped2", np.asarray(crop_frame2))
            # cv2.imshow("cropped_gray", gray)

            crop_frame_gray = gray[y:y + h, x:x + w]

            new_image_resized_gray = normalize_face_size(crop_frame_gray)
            crop_gray_array = np.asarray(new_image_resized_gray)
            # print crop_gray_array[50,50]

            cv2.imshow("little_cropped_gray", np.asarray(new_image_resized_gray))
            cv2.imwrite(os.path.join(dirname, 'saving' + temps + '.pgm'), np.asarray(new_image_resized_gray))
            # new_image=normalize_image_for_face_detection(crop_frame)
            #cv2.imshow("normalize", new_image)

        elif ch == 114:  # if R pressed
            crop_frame = frame[y:y + h, x:x + w]
            cv2.imshow("cropped", crop_frame)

            crop_frame_gray = gray[y:y + h, x:x + w]

            new_image_resized_gray = normalize_face_size(crop_frame_gray)
            predicted_label = predict_image_from_model(model, np.asarray(new_image_resized_gray))
            print 'Personne reconnue: %(predicted)s, label: %(label)s ' % {
            "predicted": corresponding[predicted_label[0]], "label": predicted_label[0]}
        elif ch == 116:
            print "pressed T"


if __name__ == '__main__':
    main()