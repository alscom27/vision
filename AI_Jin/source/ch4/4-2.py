import numpy as np #다차원 배열을 쉽게 처리하고 효율적으로 사용할 수 있도록지원하는 파이썬의 패키지
import tensorflow as tf  #텐서플로우 패키지
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import dlib

# 테스트 제네레이터 만들기
test_generator = ImageDataGenerator(rescale=1/255)
test_dataset = test_generator.flow_from_directory(directory='fer2013/test',
                                                  target_size=(48, 48),
                                                  class_mode='categorical',
                                                  batch_size=1,
                                                  shuffle=False,
                                                  seed=10)

network = load_model('emotion_best.h5')



image_list = [] 
test_images = os.path.join(os.getcwd(), "img") #이미지를 가져올 폴더 지정

#CNN 얼굴 디텍터를 사용해서 정확하게 얼굴 검출(detection) 수행
face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')


for root, dirs, files in os.walk(test_images) : #파일 목록 가져오기
    for file in files :
        if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("png") : #이미지 파일 필터링
            image_path = os.path.join(test_images, file)
            print(image_path)
            image_list.append(cv2.imread(image_path))

for img in image_list : #가져온 이미지들에 대해 진행
    #gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #흑백으로 변환
    #faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5) #얼굴 인식
    faces = face_detector(img, 1)
    for face_detection in faces:
        left, top = face_detection.rect.left(), face_detection.rect.top()
        right, bottom =  face_detection.rect.right(), face_detection.rect.bottom()
        roi = img[top:bottom, left:right]
        roi = cv2.resize(roi, (48, 48))
        roi = roi / 255
        roi = np.expand_dims(roi, axis=0)
        pred_probability = network.predict(roi)
        print(pred_probability)
        print(np.argmax(pred_probability))    
        print(test_dataset.class_indices)
        i = 0
        name = ""
        for index in test_dataset.class_indices:
            print(index)
            if i == np.argmax(pred_probability):
                name = index
            i += 1
        font = cv2.FONT_HERSHEY_SIMPLEX #폰트 지정
        cv2.putText(img, name, (left,top), font, 1, (0,0,255), 2)
        cv2.rectangle(img,(left,top),(right,bottom),(0,255,0),2)

    cv2.imshow('Preview',img) #이미지 보여주기
    if cv2.waitKey(0) >= 0:
        continue
    