import numpy as np
import cv2 as cv
import sys


def draw_Opticalflow(img, flow, step=16):
    for y in range(step // 2, frame.shape[0], step):
        for x in range(step // 2, frame.shape[1], step):
            dx, dy = flow[y, x].astype(int)

            if (dx * dx + dy * dy) > 1:
                cv.line(
                    img, (x, y), (x + dx, y + dy), (0, 0, 255), 2
                )  # 큰 모션 있는 곳은 빨간색
            else:
                cv.line(img, (x, y), (x + dx, y + dy), (0, 255, 0), 2)


cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened():
    sys.exit("카메라 연결 실패")

prev = None

while 1:
    ret, frame = cap.read()
    if not ret:
        sys.exit("프레임 획득에 실패하여 루프를 나갑니다.")

    if prev is None:
        prev = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        continue

    curr = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    draw_Opticalflow(frame, flow)
    cv.imshow("Optical flow", frame)

    prev = curr
    key = cv.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
