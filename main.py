# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 02:17:44 2023

@author: payam
"""

# real-time face recognition
from __future__ import print_function
import time
import cv2
import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse
import imutils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100,
	help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
	help="Whether or not frames should be displayed")
args = vars(ap.parse_args())






#INITIALIZE
facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_done_4classes.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
model = pickle.load(open("svm_model_160x160.pkl", 'rb'))


vs = WebcamVideoStream(src="http://192.168.1.100:4747/video").start()
fps = FPS().start()
# WHILE LOOP

ptime = 0
index = 0
while fps._numFrames < args["num_frames"]:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    for x,y,w,h in faces:
        img = rgb_img[y:y+h, x:x+w]
        img = cv.resize(img, (160,160)) # 1x160x160x3
        img = np.expand_dims(img,axis=0)
        ypred = facenet.embeddings(img)
        print(ypred)
        
        face_name = model.predict(ypred)
        print(face_name)
        final_name = encoder.inverse_transform(face_name)[0]
        cv.rectangle(frame, (x,y), (x+w,y+h), (200,255,0), 10)
        cv.putText(frame, str(final_name), (x,y-10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (200,100,255), 3, cv.LINE_AA)
    ctime = time.time()
    if ctime-ptime != 0:
        fps_ = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(frame,f' FPS :{int(fps_)}',(20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3 )
    cv.imshow("Face Recognition:", frame)

    
    
    if cv2.waitKey(1)==13:
            break
    
fps.stop()    
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))   
cv2.destroyAllWindows()
vs.stop()
