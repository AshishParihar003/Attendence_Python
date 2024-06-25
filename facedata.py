import cv2 as cv
import pickle
import numpy as np
import os
vedio = cv.VideoCapture(0)
facedetect = cv.CascadeClassifier('data/facealgo.xml')

face_data= []

i=0

name= input("Enter Your Name: ")

while True:
    ret,frame=vedio.read()
    gray= cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        crop_img= frame[y:y+h, x:x+w, :] 
        resized_img= cv.resize(crop_img, (50,50))
        if len(face_data)<=100 and i%10==0:
            face_data.append(resized_img)
        i=i+1
        cv.putText(frame,str(len(face_data)),(50,50), cv.FONT_HERSHEY_COMPLEX,1,(50,50,255),1)
        cv.rectangle(frame,(x,y),(x+y , y+h), (50,50,255), 1)
    cv.imshow("frames",frame)
    k=cv.waitKey(1)
    if k==ord('q') or len(face_data)==100:
        break

vedio.release()
cv.destroyAllWindows()

face_data= np.asanyarray(face_data)
face_data= face_data.reshape(100,-1)





if 'names.pkl' not in os.listdir('data/' ):
    names= [name]*100
    with open('data/names.pkl','wb') as f:
        pickle.dump(names,f)
        
else: 
    with open('data/names.pkl', 'rb') as f:
        names= pickle.load(f)
    names= names+[name]*100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names,f)
        
if 'faces_data.pkl' not in os.listdir('data/' ):

    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(face_data,f)
        
else: 
    with open('data/faces_data.pkl', 'rb') as f:
        faces= pickle.load(f)
    faces= np.append(faces,face_data,axis=0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(names,f) 