import streamlit as st
import cv2
import time
import tempfile
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

st.title("🔥 화재/연기 실시간 탐지 시스템 (YOLOv8 앙상블 실험)")

# 🚀 모델 경로 리스트 정의
model_options = {
    "YOLOv8n (불/연기)": "runs/firesmoke_yolov8s_tuned/weights/best.pt",
    "YOLOv8n (연기 전용)": "runs/smoke_detect_v8n/weights/best.pt",
    "YOLOv8n-seg (불/연기)": "runs/firesmoke_seg_train/weights/best.pt",
    "YOLOv8s (병합)": "runs/merge_detect_v8s/weights/best.pt",
    "YOLOv8s (연기 전용)": "runs/smoke_detect_v8s/weights/best.pt",
    "YOLOv8s-seg": "runs/firesmoke_seg_s/weights/best.pt",
    "YOLO11n-seg": "runs/segment/firesmoke_seg_v11/weights/best.pt",
    "YOLO11n-seg-smoke": "runs/segment/firesmoke_seg_v11_smoke_focus/weights/best.pt",
    "YOLOv8s (최고 성능)": "runs/detect/firesmoke_detect_v8s/weights/best.pt",
}

selected_models = st.multiselect(
    "✅ 사용할 모델 조합을 선택하세요",
    list(model_options.keys()),
    default=["YOLOv8n (불/연기)", "YOLOv8n (연기 전용)", "YOLOv8n-seg (불/연기)"],
)

models = {name: YOLO(model_options[name]) for name in selected_models}
FRAME_WINDOW = st.empty()
option = st.radio("🎥 입력 소스 선택", ["웹캠", "영상 업로드"])
TEST_DURATION = 60
frame_count, inference_times = 0, []


# 🎯 박스 필터링
def filter_boxes(boxes, scores, labels, min_area=0.001, max_area=0.4, conf_thr=0.3):
    filtered = []
    for box, score, label in zip(boxes, scores, labels):
        if score < conf_thr:
            continue
        w, h = box[2] - box[0], box[3] - box[1]
        area = w * h
        if min_area <= area <= max_area:
            filtered.append((box, score, label))
    return zip(*filtered) if filtered else ([], [], [])


# 📦 앙상블
def ensemble_predictions(predictions, iou_thr=0.5, skip_box_thr=0.001):
    boxes, scores, labels = [], [], []
    for pred in predictions:
        b, s, l = [], [], []
        for box, score, label in zip(
            pred.boxes.xyxy.cpu().numpy(),
            pred.boxes.conf.cpu().numpy(),
            pred.boxes.cls.cpu().numpy(),
        ):
            x1, y1, x2, y2 = box
            w, h = pred.orig_shape[1], pred.orig_shape[0]
            b.append([x1 / w, y1 / h, x2 / w, y2 / h])
            s.append(float(score))
            l.append(int(label))
        boxes.append(b)
        scores.append(s)
        labels.append(l)

    boxes, scores, labels = weighted_boxes_fusion(
        boxes, scores, labels, iou_thr=iou_thr, skip_box_thr=skip_box_thr
    )
    return filter_boxes(boxes, scores, labels)


# 🖼️ 박스 시각화
def draw_boxes(frame, boxes, scores, labels):
    h, w, _ = frame.shape
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, [box[0] * w, box[1] * h, box[2] * w, box[3] * h])
        color = (0, 0, 255) if label == 0 else (0, 255, 255)
        name = ["fire", "smoke"][int(label)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{name} {score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return frame


# 🎨 마스크 시각화
def draw_masks(frame, masks, color=(0, 0, 255), alpha=0.4):
    if masks is None or len(masks) == 0:
        return frame

    h, w = frame.shape[:2]
    for mask in masks:
        mask = mask.cpu().numpy().astype(np.uint8)
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        colored_mask = np.zeros_like(frame, dtype=np.uint8)
        colored_mask[mask_resized > 0.5] = color
        frame = cv2.addWeighted(colored_mask, alpha, frame, 1 - alpha, 0)
    return frame


# 🎬 영상 처리
def process_video(cap, limit_time=60):
    global frame_count, inference_times
    frame_count, inference_times = 0, []
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=15)

        t0 = time.time()
        preds = [
            model(frame, conf=0.3, iou=0.3, verbose=False)[0]
            for model in models.values()
        ]
        boxes, scores, labels = ensemble_predictions(preds)

        # 🔍 seg 모델 하나만 선택해 mask 추출
        seg_masks = None
        for name, model in models.items():
            if "seg" in name:
                seg_result = model(frame, conf=0.3, iou=0.3, verbose=False)[0]
                seg_masks = seg_result.masks.data if seg_result.masks else None
                break

        t1 = time.time()

        annotated = frame.copy()
        annotated = draw_masks(annotated, seg_masks)  # 🎯 마스크 먼저
        annotated = draw_boxes(annotated, boxes, scores, labels)

        FRAME_WINDOW.image(annotated, channels="BGR", use_container_width=True)

        inference_times.append(t1 - t0)
        frame_count += 1
        if time.time() - start_time > limit_time:
            break

    cap.release()
    return time.time() - start_time


# 📥 입력 소스 처리
if option == "웹캠":
    if st.checkbox("▶️ 웹캠 시작"):
        cap = cv2.VideoCapture(0)
        elapsed_time = process_video(cap, TEST_DURATION)
elif option == "영상 업로드":
    file = st.file_uploader("📁 영상 업로드", type=["mp4", "avi", "mov"])
    if file:
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(file.read())
        cap = cv2.VideoCapture(temp.name)
        elapsed_time = process_video(cap, TEST_DURATION)

# 📊 결과 출력
if frame_count > 0:
    avg_fps = frame_count / elapsed_time
    avg_inf_time = sum(inference_times) / len(inference_times)
    infer_fps = 1 / avg_inf_time if avg_inf_time > 0 else 0

    st.markdown("## 📊 분석 결과")
    st.write(f"🔁 총 프레임 수: {frame_count}")
    st.write(f"⏱️ 총 시간: {elapsed_time:.2f}s")
    st.write(f"📸 평균 FPS: {avg_fps:.2f}")
    st.write(f"🧠 평균 추론 시간: {avg_inf_time:.4f}s/frame")
    st.write(f"⚡ 순수 추론 기준 FPS: {infer_fps:.2f}")
