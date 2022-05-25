import os
from datetime import datetime
import cv2
import face_recognition
import numpy as np
import pyttsx3


def imgLibraryImport():
    path = 'photos'
    photos = []
    global classNames
    classNames = []

    # all photos paths
    myList = os.listdir(path)

    # insert all photos to cv2 structure
    for cls in myList:
        curimg = cv2.imread(f"{path}/{cls}")
        photos.append(curimg)

        # creat list of names we know to compare
        classNames.append(os.path.splitext(cls)[0])

    # list of knows faces encodings
    global encodeListKnown
    encodeListKnown = findEncodings(photos)

    # process Indicator
    print('Encoding Complete')


def findEncodings(images):
    encodelist = []

    for img in images:
        # face_recognition function need to receive a RGB img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # use face_recognition lib to find encodings
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)

    return encodelist


# make recognition list
def markAttendance(name):
    with open('attandance.csv', 'r+') as f:
        # data stored now in the csv file
        myDataList = f.readlines()

        # store names that already shown
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            # store time for documentation
            now = datetime.now()
            dtStr = now.strftime('%H:%M:%S')
            f.writelines(f'\n {name},{dtStr}')


# simple function to say "text"
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    engine.stop()


# main function
# - open computer webcam. compare detected faces to knows faces
# - for every face show a rectangle around it
# - when find a knows face, say hello to that person
def videoWebcamRecognition():
    # known faces
    imgLibraryImport()

    # open computer webcam
    cap = cv2.VideoCapture(0)

    while True:
        # capture real time img from webcam
        success, img = cap.read()

        # reduce img size for a smaller time complexity
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)

        # find the location of the face in the reduced img
        facecCurFrame = face_recognition.face_locations(imgS)

        # encode the faces in the reduced img
        encodeCurFrame = face_recognition.face_encodings(imgS, facecCurFrame)


        for encodeFace, faceLoc in zip(encodeCurFrame, facecCurFrame):

            #faces in the img that we know who they are
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)

            #calculat the match percentage for every match found
            facedist = face_recognition.face_distance(encodeListKnown, encodeFace)

            #implement into an array
            matchIndex = np.argmin(facedist)

            # if the face is the person we think it is
            if matches[matchIndex]:

                #name of the person we found
                name = classNames[matchIndex].upper()

                #call that person
                speak(f"hello {name}")
            else:
                name = "unknown person"

            #cordinates for the rectangle
            y1, x2, y2, x1 = faceLoc

            #multiply by 4 to return to the original img size
            y1, x1, y2, x2 = y1 * 4, x1 * 4, y2 * 4, x2 * 4

            #make a rectangle with a name in it
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 10, y2 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

    #show the result in the webcam feed
    cv2.imshow('webcam', img)

    #deley the process
    cv2.waitKey(3)



########################
#        main          #
########################
videoWebcamRecognition()
