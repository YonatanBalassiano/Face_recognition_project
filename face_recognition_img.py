import face_recognition
import cv2
import numpy

#import photos for Comparison
imgbill1 = face_recognition.load_image_file('photos/bill1.jpg')
imgbill1 = cv2.cvtColor(imgbill1,cv2.COLOR_BGR2RGB)

imgbill2 = face_recognition.load_image_file('photos/bill2.jpg')
imgbill2 = cv2.cvtColor(imgbill2,cv2.COLOR_BGR2RGB)

#find the face location in photo'bill2' and covert it to visual
faceLoc = face_recognition.face_locations(imgbill1)[0]
encodeBill = face_recognition.face_encodings(imgbill1)[0]
cv2.rectangle(imgbill1,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

#find the face location in photo'bill2' and covert it to visual

faceLoc2 = face_recognition.face_locations(imgbill2)[0]
encodeBill2 = face_recognition.face_encodings(imgbill2)[0]
cv2.rectangle(imgbill2,(faceLoc2[3],faceLoc2[0]),(faceLoc2[1],faceLoc2[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeBill],encodeBill2)
face_dist = face_recognition.face_distance([encodeBill],encodeBill2)
print(results,face_dist)
cv2.putText(imgbill1,f"{results}{round(face_dist[0],2)}" ,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
cv2.imshow('bill gates', imgbill1)
cv2.imshow('bill gates2', imgbill2)
cv2.waitKey(0)


