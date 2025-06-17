import streamlit as st
import cv2
import time
import tempfile
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from collections import defaultdict

# 페이지 레이아웃을 와이드 모드로 설정
st.set_page_config(layout="wide")
# 앱 제목 표시
st.title("🔥 화재/연기 실시간 탐지 시스템 (YOLOv8 앙상블 실험)")

# 사용할 수 있는 모델들과 경로를 딕셔너리로 정의
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

# Streamlit 멀티셀렉트 위젯으로 모델 선택 허용
selected_models = st.multiselect(
    "✅ 사용할 모델 조합을 선택하세요",
    list(model_options.keys()),
    default=["YOLOv8s (최고 성능)"],
)
# 선택된 모델 로드 (YOLO 객체 생성)
models = {name: YOLO(model_options[name]) for name in selected_models}

# 비디오 출력용 빈 공간 (placeholder)
FRAME_WINDOW = st.empty()
# 입력 소스 선택: 웹캠 또는 업로드된 영상
option = st.radio("🎥 입력 속성 선택", ["웹캠", "영상 업로드"])
# 테스트 지속 시간 (초)
TEST_DURATION = 60
# 전역 변수 초기화: 프레임 수, 추론 시간, 로그
frame_count, inference_times = 0, []
detection_log, alert_log = [], []
first_alert_image = None

# 카메라 위치 정보: 내부 코드와 친숙한 이름(주방/거실/창고)
LOCATION_INFO = {
    "cam01": {"type": "주방", "allow_fire": True},
    "cam02": {"type": "거실", "allow_fire": False},
    "cam03": {"type": "창고", "allow_fire": False},
    "cam04": {"type": "작업실", "allow_fire": True},
}
# 사용자에게는 '주방 (cam01)' 형식으로 보여주고, 내부적으로 cam01 등 키를 사용
display_options = {
    f"{info['type']} ({key})": key for key, info in LOCATION_INFO.items()
}
selected_display = st.selectbox("📍 카메라 위치 선택", list(display_options.keys()))
# 선택된 값을 내부 키로 변환
cam_id = display_options[selected_display]
location_name = LOCATION_INFO[cam_id]["type"]  # 예: "주방"
allow_fire = LOCATION_INFO[cam_id]["allow_fire"]

# 세션 상태 초기화 (이전 화재 면적, 좌표별 누적 카운터, 이전 연기 농도)
if "prev_fire_area" not in st.session_state:
    st.session_state.prev_fire_area = 0
if "location_fire_counter" not in st.session_state:
    st.session_state.location_fire_counter = defaultdict(int)
if "prev_smoke_intensity" not in st.session_state:
    st.session_state.prev_smoke_intensity = {}

# 탭 구성: 전체 탐지 로그, 경고 대시보드
tab1, tab2 = st.tabs(["📋 전체 탐지 로그", "🚨 경고 대시보드"])
with tab1:
    log_table = st.empty()
with tab2:
    alert_table = st.empty()


def filter_boxes(boxes, scores, labels, min_area=0.001, max_area=0.4, conf_thr=0.3):
    """
    박스 필터링 함수:
    - 신뢰도(conf_thr) 이상
    - 박스 면적(min_area, max_area) 사이인 것만 반환
    """
    filtered = []
    for box, score, label in zip(boxes, scores, labels):
        if score < conf_thr:
            continue
        w, h = box[2] - box[0], box[3] - box[1]
        area = w * h
        if min_area <= area <= max_area:
            filtered.append((box, score, label))
    return zip(*filtered) if filtered else ([], [], [])


def ensemble_predictions(predictions, iou_thr=0.5, skip_box_thr=0.001):
    """
    모델 앙상블 함수:
    - 여러 모델(predictions)에서 출력된 박스, 스코어, 클래스 정보를
      weighted_boxes_fusion 알고리즘으로 병합 후 반환
    """
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


def draw_boxes(frame, boxes, scores, labels):
    """
    탐지된 박스를 이미지에 그려주는 함수:
    - 불(fire)은 빨강, 연기(smoke)는 노랑 색상
    - 클래스 이름과 스코어를 텍스트로 표시
    """
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


def draw_masks(frame, masks, color=(0, 0, 255), alpha=0.4):
    """
    분할(segmentation) 마스크를 이미지에 오버레이하는 함수:
    - 마스크 영역을 반투명으로 덮어줌
    """
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


def process_video(cap, limit_time=60):
    global frame_count, inference_times, detection_log, alert_log, first_alert_image
    frame_count, inference_times, detection_log, alert_log = 0, [], [], []
    start_time = time.time()
    first_alert_time = None
    first_alert_reason = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 크기 및 밝기/대비 보정
        frame = cv2.resize(frame, (640, 360))
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=15)
        # 그레이스케일로 변환해 smoke 농도 계산에 사용
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 추론 시작 시간 기록
        t0 = time.time()
        preds = [
            model(frame, conf=0.3, iou=0.3, verbose=False)[0]
            for model in models.values()
        ]
        boxes, scores, labels = ensemble_predictions(preds)

        # 분할 모델 중 마스크 추출
        seg_masks = None
        for name, model in models.items():
            if "seg" in name:
                seg_result = model(frame, conf=0.3, iou=0.3, verbose=False)[0]
                seg_masks = seg_result.masks.data if seg_result.masks else None
                break

        t1 = time.time()

        # 시각화
        annotated = frame.copy()
        annotated = draw_masks(annotated, seg_masks)
        annotated = draw_boxes(annotated, boxes, scores, labels)

        now = time.strftime("%H:%M:%S", time.localtime())
        current_fire_area = sum(
            (box[2] - box[0]) * (box[3] - box[1])
            for box, label in zip(boxes, labels)
            if label == 0
        )
        growth_rate = (
            current_fire_area / st.session_state.prev_fire_area
            if st.session_state.prev_fire_area > 0
            else 1
        )
        st.session_state.prev_fire_area = current_fire_area

        for box, score, label in zip(boxes, scores, labels):
            class_name = "fire" if label == 0 else "smoke"
            coord = tuple(round(c, 1) for c in box)
            coord_key = str(coord)

            # 연기 농도 증가율 계산
            intensity_growth = 1.0
            if class_name == "smoke":
                h_gray, w_gray = gray_frame.shape
                x1, y1, x2, y2 = [
                    int(c * dim)
                    for c, dim in zip(box, (w_gray, h_gray, w_gray, h_gray))
                ]
                roi = gray_frame[y1:y2, x1:x2]
                if roi.size > 0:
                    mean_intensity = float(np.mean(roi))
                    prev_int = st.session_state.prev_smoke_intensity.get(
                        coord_key, mean_intensity
                    )
                    intensity_growth = mean_intensity / (prev_int + 1e-6)
                    st.session_state.prev_smoke_intensity[coord_key] = mean_intensity

            # 탐지 로그 등록
            detection_log.append(
                {
                    "시간": now,
                    "클래스": class_name,
                    "신뢰도": round(score, 2),
                    "좌표": str(coord),
                }
            )

            # 경고 판단
            reason = ""
            is_fire_risk = False
            if class_name == "fire":
                if not allow_fire and score >= 0.6:
                    is_fire_risk = True
                    reason = "허용되지 않은 위치에서 신뢰도 ≥ 0.6"
                elif not allow_fire and growth_rate > 1.5 and score >= 0.5:
                    is_fire_risk = True
                    reason = "면적 성장률 > 1.5 및 신뢰도 ≥ 0.5"
                elif growth_rate > 3 and score >= 0.7:
                    is_fire_risk = True
                    reason = "면적 성장률 > 3 및 신뢰도 ≥ 0.7"
                else:
                    cnt = st.session_state.location_fire_counter[coord_key] + 1
                    st.session_state.location_fire_counter[coord_key] = cnt
                    if not allow_fire and cnt >= 5:
                        is_fire_risk = True
                        reason = "반복 탐지 카운터 ≥ 5"
            elif class_name == "smoke":
                if score >= 0.7:
                    is_fire_risk = True
                    reason = "연기 신뢰도 ≥ 0.7"
                elif intensity_growth > 1.2:
                    is_fire_risk = True
                    reason = f"연기 농도 증가율 > 1.2 (x{intensity_growth:.2f})"

            if is_fire_risk:
                alert_log.append(
                    {
                        "⏰ 시간": now,
                        "⚠️ 위험": class_name,
                        "신뢰도": round(score, 2),
                        "농도증가율": round(intensity_growth, 2),
                        "좌표": str(coord),
                    }
                )
                if first_alert_time is None:
                    first_alert_time = time.time() - start_time
                    first_alert_image = annotated.copy()
                    first_alert_reason = reason

        # 로그 크기 유지
        if len(detection_log) > 100:
            detection_log = detection_log[-100:]
        if len(alert_log) > 50:
            alert_log = alert_log[-50:]

        # 테이블 업데이트
        log_df = pd.DataFrame(detection_log)
        alert_df = pd.DataFrame(alert_log)
        with tab1:
            log_table.dataframe(log_df, use_container_width=True)
        with tab2:
            alert_table.dataframe(alert_df, use_container_width=True)

        FRAME_WINDOW.image(annotated, channels="BGR", use_container_width=True)

        inference_times.append(t1 - t0)
        frame_count += 1
        if time.time() - start_time > limit_time:
            break

    cap.release()
    return (
        time.time() - start_time,
        first_alert_time,
        first_alert_image,
        first_alert_reason,
    )


# 입력 소스별 처리
if option == "웹캠":
    if st.checkbox("▶️ 웹캠 시작"):
        cap = cv2.VideoCapture(0)
        elapsed_time, first_alert_time, first_alert_image, first_alert_reason = (
            process_video(cap, TEST_DURATION)
        )
elif option == "영상 업로드":
    file = st.file_uploader("📁 영상 업로드", type=["mp4", "avi", "mov"])
    if file:
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(file.read())
        cap = cv2.VideoCapture(temp.name)
        elapsed_time, first_alert_time, first_alert_image, first_alert_reason = (
            process_video(cap, TEST_DURATION)
        )

# 분석 결과 출력
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

    if first_alert_time is not None:
        st.markdown(f"🚨 최초 화재 경고 시점: **{first_alert_time:.2f}초**")
        st.markdown(f"**최초 경고 이유:** {first_alert_reason}")
        st.image(
            cv2.cvtColor(first_alert_image, cv2.COLOR_BGR2RGB),
            caption="🚨 최초 경고 시점 이미지",
            use_container_width=True,
        )
