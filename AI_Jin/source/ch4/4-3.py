import pandas as pd #데이터를 분석 및 조작하기 위한 소프트웨어 라이브러리
import numpy as np #다차원 배열을 쉽게 처리하고 효율적으로 사용할 수 있도록지원하는 파이썬의 패키지
import matplotlib.pyplot as plt #다양한 데이터를 많은 방법으로 도식화 할 수 있도록 하는 파이썬 라이브러리
import tensorflow as tf  #텐서플로우 패키지
from keras.models import load_model


import cv2
import dlib


network = load_model('emotion_best.h5')

cap = cv2.VideoCapture('test.mp4')

video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 프레임 넓이
video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 프레임 높이
video_size = (round(video_width), round(video_height)) # 비디오 사이즈
video_fps = cap.get(cv2.CAP_PROP_FPS)  # FPS(Frames Per Second)
frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 프레임의 수

print('Number of frames:', frame_cnt, '/ FPS:', round(video_fps), '/ Frame size:', video_size)

video_output_path = 'emotion_classification_result.avi'
codec = cv2.VideoWriter_fourcc(*'XVID')  # Set the codec  
video_writer = cv2.VideoWriter(video_output_path, codec, video_fps, video_size)


cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

green_color=(0, 255, 0)
red_color=(0, 0, 255)
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

while (cv2.waitKey(1) < 0):
    connected, frame = cap.read()  # Read one frame from a VideoCapture object
    if not connected:
        break
    face_detections = cnn_face_detector(frame, 1)
    if len(face_detections) > 0:
        for face_detection in face_detections:
            left, top = face_detection.rect.left(), face_detection.rect.top()
            right, bottom, confidence = face_detection.rect.right(), face_detection.rect.bottom(), face_detection.confidence
            cv2.rectangle(frame, (left, top), (right, bottom), green_color, 2)
            roi = frame[top:bottom, left:right]
            roi = cv2.resize(roi, (48, 48))  # Extract region of interest from image
            roi = roi / 255  # Normalize
            roi = np.expand_dims(roi, axis=0)
            preds = network.predict(roi)

            if preds is not None:
                pred_emotion_index = np.argmax(preds)
                cv2.putText(frame, emotions[pred_emotion_index], (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, 1)
    
    video_writer.write(frame)

video_writer.release()
cap.release()