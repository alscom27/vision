from deepface import DeepFace
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace
import cv2, glob
import json

img_list = glob.glob('img/*.jpg')

print(img_list)

for img_path in img_list:
    img = cv2.imread(img_path)
    result = DeepFace.analyze(img_path, actions = ['age', 'gender', 'race', 'emotion'])
    print(result['age'])
    
    gender  = result['gender']
    age = result['age']
    emotion = result['dominant_emotion']
    race = result['dominant_race']
    x1 = result['region']['x']
    y1 = result['region']['y']

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

