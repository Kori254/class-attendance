import cv2
import mediapipe as mp
import numpy as np
import pickle
import cv2

recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizers/face-trainner.yml')

labels={"person_name":1}
with open("pickles/face-labels.pickle",'rb') as f:
    og_labels=pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap=cv2.VideoCapture("me.mp4")

facedetect=mp.solutions.face_detection
drawing=mp.solutions.drawing_utils
face=facedetect.FaceDetection(0.75)


while True:
    ret,frame=cap.read()
    image=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results=face.process(gray)
    
    if results.detections:
        for id,detection in enumerate(results.detections):
            BBOC=detection.location_data.relative_bounding_box
            ih,iw,ic=frame.shape
            bbox= int(BBOC.xmin*iw),int(BBOC.ymin*ih),\
                int(BBOC.width*iw),int(BBOC.height*ih)
            cv2.rectangle(frame,bbox,(255,0,255))
            print(bbox)
            x,y,w,h=bbox[0],bbox[1],bbox[2],bbox[3]
            roi=image[y:y+h,x:x+w]
            id_,conf=recognizer.predict(roi)
            if conf<100:
                print(id_)
                print(labels[id_])

            else:
                print('uknown')

                  
    cv2.imshow('recognizer',frame)
    k = cv2.waitKey(27) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()    