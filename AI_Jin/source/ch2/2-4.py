import cv2
import dlib
import math
import time

# 이미지를 로딩함!
image = cv2.imread('bts.jpg')
image_resized = cv2.resize(image, (755, 500))
image_resized_CNN = image_resized.copy()

#HOG 검출기로 얼굴을 인식해보자
start = time.time()
hog_face_detector = dlib.get_frontal_face_detector()
face_detections = hog_face_detector(image_resized, 1)
end = time.time()

for face_detection in face_detections:
    left, top, right, bottom = face_detection.left(), face_detection.top(), face_detection.right(), face_detection.bottom()
    cv2.rectangle(image_resized, (left, top), (right, bottom), (0, 255, 0), 2)

print(f"{end - start:.5f} HOG 검출 걸린 초")
#dlib의 cnn_face_detection_model_v1() 메서드로 CNN을 활용한 검출기를 불러올 수 있음

start = time.time()
cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
face_detections_CNN = cnn_face_detector(image_resized_CNN, 1)
end = time.time()
print(f"{end - start:.5f} CNN 검출 걸린 초")


# 캡쳐를 하기에 칸이 좁아서 5개 변수를 다음과 같이 3개 2개로 나눔
for idx, face_detection_CNN in enumerate(face_detections_CNN):
    left, top, right = face_detection_CNN.rect.left(), face_detection_CNN.rect.top(), face_detection_CNN.rect.right(), 
    bottom, confidence = face_detection_CNN.rect.bottom(), face_detection_CNN.confidence
    print(f'confidence{idx+1}: {confidence}')  # print confidence of the detection
    cv2.rectangle(image_resized_CNN, (left, top), (right, bottom), (0, 255, 0), 2)


cv2.imshow("HOG", image_resized)
cv2.imshow("CNN", image_resized_CNN)
cv2.waitKey(0)
cv2.destroyAllWindows()


