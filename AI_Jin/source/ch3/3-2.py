import cv2
import numpy as np
import os
from PIL import Image
import dlib #hog 인식을 위한 dlib 라이브러리 추가

labels = ["Cha", "Ma"] #라벨 지정
 
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# HOG 인식으로 변경하기
hog_face_detector = dlib.get_frontal_face_detector()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-trainner.yml") #저장된 값 가져오기
 
image_list = [] 

test_images = os.path.join(os.getcwd(), "test_list") #이미지를 가져올 폴더 지정

for root, dirs, files in os.walk(test_images) : #파일 목록 가져오기
    for file in files :
        if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("png") : #이미지 파일 필터링
            image_path = os.path.join(test_images, file)
            print(image_path)
            image_list.append(cv2.imread(image_path))

for img in image_list : #가져온 이미지들에 대해 진행
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #흑백으로 변환
    #faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5) #얼굴 인식
    faces = hog_face_detector(gray, 1)
    #for (x, y, w, h) in faces:
    for face_detection in faces:
        x, y, x2, y2 = face_detection.left(), face_detection.top(), face_detection.right(), face_detection.bottom()
        roi = gray[y:y2, x:x2] #얼굴부분만 가져오기
        id_, conf = recognizer.predict(roi) #얼마나 유사한지 확인
        print(labels[id_], conf)
        if conf>=50:
            font = cv2.FONT_HERSHEY_SIMPLEX #폰트 지정
            name = labels[id_] #ID를 이용하여 이름 가져오기
            cv2.putText(img, name, (x,y), font, 1, (0,0,255), 2)
            cv2.rectangle(img,(x,y),(x2,y2),(0,255,0),2)
    
    cv2.imshow('Preview',img) #이미지 보여주기
    if cv2.waitKey(0) >= 0:
        continue
cv2.destroyAllWindows()