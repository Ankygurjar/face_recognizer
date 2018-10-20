import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #For Detecting Face

#For Capturing Face using webcam

cam = cv2.VideoCapture(0)

#For capturing Frame
#For Naming the face captured

id = input("Enter user id")
sampleNum = 0

while(True):
    ret,img = cam.read();
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #For converting the colored image into gray image

    faces = faceDetect.detectMultiScale(gray,1.3,5)

    #For Multiple faces
    for(x,y,w,h) in faces:
        sampleNum = sampleNum+1
        
        #for capturng the faces and storing
        cv2.imwrite("dataSet/user."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.waitKey(100)
        
    cv2.imshow("Face",img)
    cv2.waitKey(1)
    if(sampleNum > 40):
        break;
    
cam.release()
cv2.destroyAllWindows()
    
