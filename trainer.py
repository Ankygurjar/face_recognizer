import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.createLBPHFaceRecognizer()#to recognize the image

path = 'dataSet'

def getImagesWithID(path):

    #for extracting the images
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]

    faces = []
    IDs = []

    for imagePath in imagePaths:

        #Converting images into numpy arrays
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg,'uint8')

        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("trainer",faceNp)
        cv2.waitKey(100)

    return np.array(IDs), faces

Ids,faces = getImagesWithID(path)

recognizer.train(faces, Ids)
recognizer.save('recognizer/trainingData.yml')
cv2.destroyAllWindows()


        
