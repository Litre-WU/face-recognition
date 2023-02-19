# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 17:41:03 2023

@author: payam
"""

import cv2
import os
import numpy as np



# dataset generating

def generate_dataset(save_directory):
    name = input("Enter username for data collection:")
    print("Please move your head into different sides for better accuracy. This will take a few minutes, please be patient.")
    cam_path = "http://192.168.1.100:4747/video" 
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    def face_cropped(img):
        # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(img, 1.3, 5)
        # scaling factor = 1.3
        # minimum neighbor = 5
        
        if faces is ():
            return None
        for (x,y,w,h) in faces:
            cropped_face = img[y:y+h,x:x+w]
        return cropped_face
    
    cap = cv2.VideoCapture(cam_path)
    id =1
    img_id = 0
    
    while True:
        ret, frame = cap.read()
        if face_cropped(frame) is not None:
            img_id+=1
            face = cv2.resize(face_cropped(frame), (200,200))
            
            file_name_path = "dataset/Unkown/"+str(name)+"."+str(img_id)+".jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            
            cv2.imshow("Cropped face", face)
            
        if cv2.waitKey(1)==13 or int(img_id)==1000: #13 is the ASCII character of Enter
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed....")
generate_dataset("dataset/Unkown")


