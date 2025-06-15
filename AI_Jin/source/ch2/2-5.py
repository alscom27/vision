import cv2
import dlib
import math
import time

webcam = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()

if not webcam.isOpened():
    print("Could not open webcam")
    exit()
while webcam.isOpened():
    status, frame = webcam.read()
    image_resized = cv2.resize(frame, (755, 500))
    #HOG 검출기로 얼굴을 인식해보자
    start = time.time()
    face_detections = hog_face_detector(image_resized, 1)
    end = time.time()
    for face_detection in face_detections:
        left, top, right, bottom = face_detection.left(), face_detection.top(), face_detection.right(), face_detection.bottom()
        cv2.rectangle(image_resized, (left, top), (right, bottom), (0, 255, 0), 2)
    org=(50,100)
    text = f"{end - start:.5f} HOGs"
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image_resized,text,org,font,1,(255,0,0),2)

    if status:
        cv2.imshow("test", image_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
webcam.release()
cv2.destroyAllWindows()