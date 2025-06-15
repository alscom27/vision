from deepface import DeepFace
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace
import cv2, glob, dlib
import json


webcam = cv2.VideoCapture(0)
frame = webcam.read()

if not webcam.isOpened():
    print("Could not open webcam")
    exit()
while webcam.isOpened():
    status, frame = webcam.read()
    if status:
        cv2.imshow("test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
webcam.release()
cv2.destroyAllWindows()

img_list = glob.glob('img/*.jpg')
print(img_list)

for img_path in img_list:
    img = cv2.imread(img_path)
    result = DeepFace.analyze(img_path, actions = ['age', 'gender', 'race', 'emotion'])
    print(result[0]['age'])
    
    
    gender  = result[0]['dominant_gender']
    age = result[0]['age']
    emotion = result[0]['dominant_emotion']
    race = result[0]['dominant_race']
    x1 = result[0]['region']['x']
    y1 = result[0]['region']['y']

    overlay_text = '%s %s %s %s' % (gender, age, emotion, race)
    cv2.putText(img, overlay_text, org=(x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
      fontScale=1, color=(0,0,0), thickness=10)
    cv2.putText(img, overlay_text, org=(x1, y1),
      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255), thickness=2)
    cv2.imshow('img', img)
    cv2.imwrite('result/%s' % img_path.split('/')[-1], img)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

