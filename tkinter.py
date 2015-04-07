__author__ = 'guillaume'

import Tkinter as tk
import os
import datetime
from PIL import Image, ImageTk

import cv2

import trackers
import recognizers


db_manager = recognizers.FaceDbManager()
db_manager.create_csv()
db_manager.create_dict()
print db_manager._faces_dictionary

width, height = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

tracker = trackers.FaceTracker(scaleFactor=1.2, minNeighbors=2, flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
eigen_recognizer = recognizers.EigenRecognizer()
fisher_recognizer = recognizers.FisherRecognizer()

fenetre = tk.Tk()

# creation de la frame de gauche
left_frame = tk.Frame(fenetre, width=width, height=height)
left_frame.pack(side=tk.LEFT)

#assignation frame cote droit
right_frame = tk.Frame(fenetre, )
right_frame.pack(side=tk.RIGHT)

#creation pour l image opencv
image_open = tk.Label(left_frame)
image_open.pack()


def show_result(label, metric, tr, type=0):
    #fonction appelee pour changer l'image dans le petit cadre et resultat et %
    pourcentage = str(metric)
    print pourcentage
    if tr == 0:  #pas temps reel
        if label == -1:
            name_result = "non reconnu"
            image_result.configure(image="")
        else:
            name_result = db_manager.LabelNameTable[int(label)]
            path = "pictures/" + name_result
            listing = os.listdir(path)
            path_img = path + "/" + listing[0]
            image = Image.open(path_img)
            photo = ImageTk.PhotoImage(image)
            image_result.configure(image=photo)
            image_result.image = photo

        print label
        #affichage de la ligne resultat
        eigen_result.configure(text="")
        fisher_result.configure(text="")
        R.configure(text="Resultat: " + name_result + " ( " + pourcentage + " )")
        #affichage de l image en bas a droite
    elif tr == 1:  #temps reel
        if label == -1:
            name_result = "non reconnu"
        else:
            name_result = db_manager.LabelNameTable[int(label)]
            R.configure(text="")
            image_result.configure(image="")
            if type == 1:
                eigen_result.configure(text="Resultat eigen: " + name_result + " ( " + pourcentage + " )")
            elif type == 2:
                fisher_result.configure(text="Resultat fisher: " + name_result + " ( " + pourcentage + " )")


def recognize_eigenfaces(frame, tr):
    #ici pour l'algo de eigenfaces
    label = -1
    metric = 0
    for face in tracker.faces:
        if face.proper_face():
            norm_face = face.normalized_face(frame)

            label, metric = eigen_recognizer.predict(norm_face)
    show_result(label, metric, tr, 1)


def recognize_fisherfaces(frame, tr):
    #ici pour l'algo de fisherfaces
    label = -1
    metric = 0
    for face in tracker.faces:
        if face.proper_face():
            norm_face = face.normalized_face(frame)

            label, metric = fisher_recognizer.predict(norm_face)
    show_result(label, metric, tr, 2)


def save_frame():
    #ici pour la fonction de sauvegarder la photo
    #il faudra remplacer cv2test par la bonne variable de la tete
    temps = datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')
    _, frame = cap.read()
    #frame = cv2.flip(frame, 1)
    #cv2test = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #permet de sauvegarder image dans le bon dossier
    name = E.get()
    found = 0
    listing = os.listdir('pictures')
    for dirname1 in listing:
        if (dirname1 == name):
            found = 1

    if found == 0:
        os.mkdir('pictures/' + name)

    dirname = "pictures/" + name
    for face in tracker.faces:
        cv2.imwrite(os.path.join(dirname, 'saving' + temps + '.pgm'), face.normalized_face(frame))


def show_frame():
    global frame
    _, frame = cap.read()
    framecolor = cv2.cvtColor(frame, cv2.cv.CV_BGR2RGBA)
    frame = cv2.cvtColor(frame, cv2.cv.CV_BGR2GRAY)
    if choix_r.get() == 2:
        tracker.drawDebugRects(framecolor)
    tracker.refresh(frame)
    if choix_tr.get() == 2:
        recognize_fisherfaces(frame, 1)
        recognize_eigenfaces(frame, 1)

    frame = cv2.flip(frame, 1)
    cv2image = framecolor
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    image_open.imgtk = imgtk
    image_open.configure(image=imgtk)
    image_open.after(10, show_frame)


#placement des differents boutons

DR = tk.Label(right_frame, text="Draw Rectangle")
DR.pack()
choix_r = tk.IntVar()
Radio3 = tk.Radiobutton(right_frame, text="Non", variable=choix_r, value=1)
Radio3.pack()
Radio4 = tk.Radiobutton(right_frame, text="Oui", variable=choix_r, value=2)
Radio4.pack()

TR = tk.Label(right_frame, text="Temps reel:")
TR.pack()
choix_tr = tk.IntVar()
Radio1 = tk.Radiobutton(right_frame, text="Non", variable=choix_tr, value=1)
Radio1.pack()
Radio2 = tk.Radiobutton(right_frame, text="Oui", variable=choix_tr, value=2)
Radio2.pack()

T = tk.Label(right_frame, text="Nom :")
T.pack()

E = tk.Entry(right_frame)
E.pack()

A = tk.Button(right_frame, text="Save Picture", command=save_frame)
A.pack()

C = tk.Button(right_frame, text="Eigenfaces", command=lambda: recognize_eigenfaces(frame, 0))
C.pack()

D = tk.Button(right_frame, text="Fisherfaces", command=lambda: recognize_fisherfaces(frame, 0))
D.pack()

B = tk.Button(right_frame, text="Quitter", command=fenetre.quit)
B.pack()

R = tk.Label(right_frame)
R.pack()

eigen_result = tk.Label(right_frame)
eigen_result.pack()

fisher_result = tk.Label(right_frame)
fisher_result.pack()

#image created called at the initialisation

image_result = tk.Label(right_frame)
image_result.pack()

show_frame()
fenetre.mainloop()
