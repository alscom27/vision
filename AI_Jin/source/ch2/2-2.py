import cv2
import numpy as np
import os
from PIL import Image

# 이미지를 로딩함!
image = cv2.imread('people.jpg')
image_resized = cv2.resize(image, (755, 500))
 
cascade_face_detector  = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_detections = cascade_face_detector.detectMultiScale(image_resized, scaleFactor=1.1, minNeighbors=4)

print(face_detections)

for (x, y, w, h) in face_detections:
    cv2.rectangle(image_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Harr", image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()