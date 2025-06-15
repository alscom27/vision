import cv2
import math
import time

labels = ["Cha", "Ma"] #라벨 지정
webcam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-trainner.yml") #저장된 값 가져오기

if not webcam.isOpened():
    print("Could not open webcam")
    exit()
while webcam.isOpened():
    status, frame = webcam.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #흑백으로 변환
    #CNN 검출기로 얼굴을 인식해보자
    start = time.time() 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5) #얼굴 인식
    end = time.time()

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w] #얼굴 부분만 가져오기
        id_, conf = recognizer.predict(roi_gray) #얼마나 유사한지 확인
        print(labels[id_], conf)
        
        if conf>=50:
            font = cv2.FONT_HERSHEY_SIMPLEX #폰트 지정
            name = labels[id_] #ID를 이용하여 이름 가져오기
            cv2.putText(frame, name, (x,y), font, 1, (0,0,255), 2)
            org=(50,100)
            text = f"{end - start:.5f} seconds"
            cv2.putText(frame,text,org,font,1,(255,0,0),2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    if status:
        cv2.imshow("test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
webcam.release()
cv2.destroyAllWindows()