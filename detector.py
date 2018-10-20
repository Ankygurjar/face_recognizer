import cv2
import numpy as np
from PIL import Image


faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #For Detecting Face

#For Capturing Face using webcam   cv2.fromarray(img),  cv2.putText(cv2.fromArray(img),str(id),(x,y+h),font,(0,0,255),2)

cam = cv2.VideoCapture(0)


rec = cv2.face.createLBPHFaceRecognizer()
rec.load('recognizer//trainingData.yml')
id = 0

fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (0, 0, 0)

#For capturing Frame
while(True):
    ret,img = cam.read();
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #For converting the colored image into gray image

    faces = faceDetect.detectMultiScale(gray,1.3,5)

    #For Multiple faces
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id,conf = rec.predict(gray[y:y+h,x:x+w])

        if(id==1):
            id = "Ankit Gurjar"
        elif(id==2):
            id = "Sonu Pandit"
        elif(id==3):
            id="Arpit Saini"
        elif(id==4):
            id="Shubham"
        elif(id==6):
            id="Mummy"
        elif(id==7):
            id="Akshay"
        elif(id==8):
            id='''Sanya Ghai
                  Age:21'''
        cv2.putText(img, str(id), (x,y+h), fontface, fontscale, fontcolor)
        
    cv2.imshow("Face",img)
    if(cv2.waitKey(1) == ord('q')):
       break;

cam.release()
cv2.destroyAllWindows()
    
