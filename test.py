from sklearn.neighbors import KNeighborsClassifier

import cv2 as cv
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speach(strl):
    speach=Dispatch(("SAPI.SpVoice"))
    speach.Speak(strl)
    
vedio = cv.VideoCapture(0)
facedetect = cv.CascadeClassifier('data/facealgo.xml')

with open('data/names.pkl', 'rb') as f:
    LABLES= pickle.load(f)
with open('data/faces_data.pkl', 'rb') as f:
    FACES= pickle.load(f)

knn= KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES,LABLES)

COL_NAME=['NAME','TIME']

while True:
    ret,frame=vedio.read()
    gray= cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        crop_img= frame[y:y+h, x:x+w, :] 
        resized_img= cv.resize(crop_img, (50,50)).flatten().reshape(1,-1)
        output= knn.predict(resized_img)
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp= datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist=os.path.isfile("Attendence/Attendencess"+ date + ".csv")
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,225),1)
        cv.rectangle(frame,(x,y),(x+w,y+h),(50,50,225),2)
        cv.rectangle(frame,(x,y-40),(x+w,y),(50,50,225),-1)
        cv.putText(frame,str(output[0]),(x,y-15), cv.FONT_HERSHEY_COMPLEX,1,(255,255,255), 1 )
        # if len(face_data)<=100 and i%10==0:
        #     face_data.append(resized_img)
        # i=i+1
        # cv.putText(frame,str(len(face_data)),(50,50), cv.FONT_HERSHEY_COMPLEX,1,(50,50,255),1)
        cv.rectangle(frame,(x,y),(x+y , y+h), (50,50,255), 1)
        attendence= [str(output[0]),str(timestamp)]
    cv.imshow("frames",frame)
    k=cv.waitKey(1)
    if k==ord('o'):
        speach("Attendence taken..")
        time.sleep(5)
        if exist:
            with open("Attendence/Attendencess"+ date + ".csv","+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(attendence)
            csvfile.close()
        else:
            with open("Attendence/Attendencess"+ date + ".csv","+a") as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(COL_NAME)
                writer.writerow(attendence)
            csvfile.close()
    if k==ord('q') :
        break

vedio.release()
cv.destroyAllWindows()

# face_data= np.asanyarray(face_data)
# face_data= face_data.reshape(100,-1)





# if 'names.pkl' not in os.listdir('data/' ):
#     names= [name]*100
#     with open('data/names.pkl','wb') as f:
#         pickle.dump(names,f)
        
# else: 
#     with open('data/names.pkl', 'rb') as f:
#         names= pickle.load(f)
#     names= names+[name]*100
#     with open('data/names.pkl', 'wb') as f:
#         pickle.dump(names,f)
        
# if 'faces_data.pkl' not in os.listdir('data/' ):

#     with open('data/faces_data.pkl', 'wb') as f:
#         pickle.dump(face_data,f)
        
# else: 
#     with open('data/faces_data.pkl', 'rb') as f:
#         faces= pickle.load(f)
#     faces= np.append(faces,face_data,axis=0)
#     with open('data/faces_data.pkl', 'wb') as f:
#         pickle.dump(names,f) 