import cv2 as cv
import mediapipe as mp

img = cv.imread('./imgs/BSDS_376001.jpg')

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
res = face_detection.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))

if not res.detections:
    print('검출 실패')
else:
    for detection in res.detections:
        mp_drawing.draw_detection(img, detection)
    cv.imshow('face detection by mediapipe', img)
    
cv.waitKey()
cv.destroyAllWindows()