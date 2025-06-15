import cv2
import dlib
import math
import time
webcam = cv2.VideoCapture(0)
cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

while webcam.isOpened():
    status, frame = webcam.read()
    image_resized = cv2.resize(frame, (755, 500))

    #CNN 검출기로 얼굴을 인식해보자
    start = time.time()

    face_detections_CNN = cnn_face_detector(image_resized, 1)
    end = time.time()
    for idx, face_detection_CNN in enumerate(face_detections_CNN):
        left, top, right, bottom, confidence = face_detection_CNN.rect.left(), face_detection_CNN.rect.top(), face_detection_CNN.rect.right(), face_detection_CNN.rect.bottom(), face_detection_CNN.confidence
        print(f'confidence{idx+1}: {confidence}')  # print confidence of the detection
        cv2.rectangle(image_resized, (left, top), (right, bottom), (0, 255, 0), 2)
    org=(50,100)
    text = f"{end - start:.5f} CNNs"
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image_resized,text,org,font,1,(255,0,0),2)
    if status:
        cv2.imshow("test", image_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
webcam.release()
cv2.destroyAllWindows()