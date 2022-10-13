import cv2
import mediapipe as mp
import numpy as np
import pickle
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

model = load_model('recognizers/keras_model.h5')

labels={"person_name":1}
with open("pickles/face-labels-1.pickle",'rb') as f:
    og_labels=pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap=cv2.VideoCapture('videos/me.mp4')

facedetect=mp.solutions.face_detection
drawing=mp.solutions.drawing_utils
face=facedetect.FaceDetection(0.75)


while True:
    ret,frame=cap.read()
    image=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    cv2.namedWindow('recognizer',cv2.WINDOW_NORMAL)    
    #cv2.resizeWindow('recognizer', 560,740)
    #frame=cv2.flip(frame,0)

    results=face.process(gray)
    my_list=[]
    if results.detections:
        for id,detection in enumerate(results.detections):
            BBOC=detection.location_data.relative_bounding_box
            #print(BBOC)
            ih,iw,ic=frame.shape
            bbox= int(BBOC.xmin*iw),int(BBOC.ymin*ih),\
                int(BBOC.width*iw),int(BBOC.height*ih)
            cv2.rectangle(frame,bbox,(255,0,255))
            #print(bbox)
            x,y,w,h=bbox[0],bbox[1],bbox[2],bbox[3]
            roi=image[y:y+h,x:x+w]
            img=gray[y:y+h,x:x+w]
            #id_,conf=recognizer.predict(roi)
            img=cv2.resize(img, (224,224))
            img=img.reshape(1, 224, 224, 3)
            prediction=model.predict(img)
            # for i in prediction:
            #     print(i)
            confidence=prediction[0][1]
            conf=round(confidence*100,2)
            
            classIndex=np.argmax(model.predict(img),axis=1)
            print(classIndex[0])
            
            face_name=classIndex[0]
            print(labels[face_name])
            my_list.append(labels[face_name])
            with open('name_list.text','w') as fp:
                fp.writelines(f'{my_list}')
            


              
    cv2.imshow('recognizer',frame)
    k = cv2.waitKey(27) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()    