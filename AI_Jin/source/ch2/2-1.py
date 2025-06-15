import cv2
import dlib
import math
import time

# 이미지를 로딩함!
image = cv2.imread('people.jpg')
image_resized = cv2.resize(image, (755, 500))

cv2.imshow("test", image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()


