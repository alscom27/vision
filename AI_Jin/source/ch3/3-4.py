from deepface import DeepFace
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace

import matplotlib.pyplot as plt
import numpy as np
import cv2

model = VGGFace.loadModel()
#model = Facenet.loadModel()
#model = OpenFace.loadModel()
#model = FbDeepFace.loadModel()

input_size = model.input_shape[1:3]

backends = ['opencv', 'ssd', 'dlib', 'mtcnn']

img1 = DeepFace.detectFace("aj1.jpg", detector_backend = backends[3])
img2 = DeepFace.detectFace("aj2.png", detector_backend = backends[3])


cv2.imshow("img1", img1)
cv2.imshow("img2", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()