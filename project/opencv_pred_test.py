import streamlit as st
import cv2
import time
import tempfile
import numpy as np
from ultralytics import YOLO

st.title("🔥 YOLOv8 화재/연기 탐지 시스템")

# 🎯 모델 선택
model_option = st.selectbox(
    "🧠 모델 선택",
    [
        "YOLOv8n (불 전용, 빠름)",
        "YOLOv8s (불 정확도↑)",
        "YOLOv8n-seg (연기/불 세그멘테이션)",
    ],
)

model_path = {
    "YOLOv8n (불 전용, 빠름)": "runs/firesmoke_detect_tuned2/weights/best.pt",
    "YOLOv8s (불 정확도↑)": "runs/firesmoke_yolov8s_tuned/weights/best.pt",
    "YOLOv8n-seg (연기/불 세그멘테이션)": "runs/firesmoke_seg_train/weights/best.pt",
}

model = YOLO(model_path[model_option])

# 입력 방식 선택
option = st.radio("🎥 입력 소스 선택", ["웹캠", "영상 업로드"])
FRAME_WINDOW = st.image([])

# 통계 변수
frame_count = 0
inference_times = []
TEST_DURATION = 30  # 테스트 시간 제한


# 🔧 마스크 후처리 함수
def is_valid_mask(mask, area_threshold=3000, center_tolerance=0.3):
    mask = mask.astype(np.uint8)
    h, w = mask.shape
    cx, cy = w // 2, h // 2
    center_roi = (
        int(cx - w * center_tolerance),
        int(cy - h * center_tolerance),
        int(cx + w * center_tolerance),
        int(cy + h * center_tolerance),
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_threshold:
            continue
        x, y, rw, rh = cv2.boundingRect(cnt)
        if (center_roi[0] <= x <= center_roi[2]) and (
            center_roi[1] <= y <= center_roi[3]
        ):
            return True
    return False


# 🎬 비디오 처리 함수
def process_video(cap, limit_time=30):
    global frame_count, inference_times
    frame_count = 0
    inference_times = []

    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        results = model(frame, conf=0.7, iou=0.5, verbose=False, retina_masks=True)[0]
        t1 = time.time()

        valid_mask_count = 0
        if results.masks is not None:
            for m in results.masks.data:
                mask_np = m.cpu().numpy()
                if is_valid_mask(mask_np):
                    valid_mask_count += 1

        if valid_mask_count > 0:
            st.warning(f"🚨 위험 감지됨! ({valid_mask_count}개 이상 유효 객체)")

        annotated = results.plot()
        FRAME_WINDOW.image(annotated, channels="BGR")

        inference_times.append(t1 - t0)
        frame_count += 1

        if time.time() - start_time >= limit_time:
            break

    cap.release()
    return time.time() - start_time


# 📸 입력 소스 처리
if option == "웹캠":
    run = st.checkbox("▶️ 웹캠 실행")
    if run:
        st.success("✅ 웹캠 실행됨. 30초 뒤 자동 종료됩니다.")
        cap = cv2.VideoCapture(0)
        elapsed_time = process_video(cap, TEST_DURATION)

elif option == "영상 업로드":
    uploaded_file = st.file_uploader("🎬 영상 파일 업로드", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        st.success("✅ 영상 분석 시작. 최대 30초 동안 분석합니다.")
        elapsed_time = process_video(cap, TEST_DURATION)

# 📊 결과 출력
if frame_count > 0:
    avg_fps = frame_count / elapsed_time
    avg_infer_time = sum(inference_times) / len(inference_times)
    infer_fps = 1 / avg_infer_time if avg_infer_time > 0 else 0

    st.markdown("## 📊 측정 결과")
    st.write(f"🔁 총 프레임 수: {frame_count}")
    st.write(f"⏱️ 총 측정 시간: {elapsed_time:.2f}초")
    st.write(f"📸 평균 FPS (전체): {avg_fps:.2f}")
    st.write(f"🧠 평균 추론 시간: {avg_infer_time:.4f}초/frame")
    st.write(f"⚡ 순수 추론 기준 FPS: {infer_fps:.2f}")
