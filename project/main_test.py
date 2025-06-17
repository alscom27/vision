import streamlit as st
import cv2
import time
import tempfile
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from collections import defaultdict

# 페이지 레이아웃 설정
st.set_page_config(layout="wide")
st.title("🔥 화재/연기 실시간 탐지 시스템 (YOLOv8 앙상블 실험)")

# 모델 옵션 정의
model_options = {
    "YOLO11n-seg-smoke": "runs/segment/firesmoke_seg_v11_smoke_focus/weights/best.pt",
    "YOLOv8s (최고 성능)": "runs/detect/firesmoke_detect_v8s/weights/best.pt",
}

# 모델 선택 및 로드
selected_model_names = st.multiselect(
    "✅ 사용할 모델 조합을 선택하세요",
    list(model_options.keys()),
    default=["YOLOv8s (최고 성능)"],
)
models = [YOLO(model_options[name]) for name in selected_model_names]

# 입력 옵션
input_source = st.radio("🎥 입력 속성 선택", ["웹캠", "영상 업로드"])

# 비디오 캡처 초기화
cap = None
if input_source == "웹캠":
    if st.checkbox("▶️ 웹캠 시작"):
        cap = cv2.VideoCapture(0)
elif input_source == "영상 업로드":
    uploaded_file = st.file_uploader("📁 영상 업로드", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(uploaded_file.read())
        cap = cv2.VideoCapture(tmp.name)

# 카메라 위치 설정
LOCATION_INFO = {
    "cam01": {"type": "주방", "allow_fire": True},
    "cam02": {"type": "거실", "allow_fire": False},
    "cam03": {"type": "창고", "allow_fire": False},
    "cam04": {"type": "작업실", "allow_fire": True},
}
display_options = {
    f"{info['type']} ({key})": key for key, info in LOCATION_INFO.items()
}
selected_display = st.selectbox("📍 카메라 위치 선택", list(display_options.keys()))
cam_id = display_options[selected_display]
allow_fire = LOCATION_INFO[cam_id]["allow_fire"]

# 세션 상태 초기화
if "prev_fire_area" not in st.session_state:
    st.session_state.prev_fire_area = 0
if "prev_smoke_intensity" not in st.session_state:
    st.session_state.prev_smoke_intensity = {}
# ★ 추가: 이전 연기 마스크 영역
if "prev_smoke_mask_area" not in st.session_state:
    st.session_state.prev_smoke_mask_area = 0

# 로그 탭 및 이미지 placeholder
tab1, tab2 = st.tabs(["📋 전체 탐지 로그", "🚨 경고 대시보드"])
log_table = tab1.empty()
alert_table = tab2.empty()
FRAME_WINDOW = st.empty()


# 바운딩박스 필터 함수
def filter_boxes(boxes, scores, labels, min_area=0.001, max_area=0.4, conf_thr=0.3):
    filtered = []
    for b, s, l in zip(boxes, scores, labels):
        if s < conf_thr:
            continue
        w, h = b[2] - b[0], b[3] - b[1]
        area = w * h
        if min_area <= area <= max_area:
            filtered.append((b, s, int(l)))
    if filtered:
        bz, sz, lz = zip(*filtered)
        return list(bz), list(sz), list(lz)
    return [], [], []


# 합의 기반 앙상블 함수
def ensemble_with_consensus(preds, min_models=2, iou_thr=0.5, skip_box_thr=0.3):
    all_boxes, all_scores, all_labels = [], [], []
    for pred in preds:
        h, w = pred.orig_shape
        b_list, s_list, l_list = [], [], []
        for box, score, label in zip(
            pred.boxes.xyxy.cpu().numpy(),
            pred.boxes.conf.cpu().numpy(),
            pred.boxes.cls.cpu().numpy(),
        ):
            x1, y1, x2, y2 = box
            b_list.append([x1 / w, y1 / h, x2 / w, y2 / h])
            s_list.append(float(score))
            l_list.append(int(label))
        all_boxes.append(b_list)
        all_scores.append(s_list)
        all_labels.append(l_list)
    boxes, scores, labels = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels, iou_thr=iou_thr, skip_box_thr=skip_box_thr
    )
    keep_b, keep_s, keep_l = [], [], []
    for box, score, label in zip(boxes, scores, labels):
        count = 0
        for b_list in all_boxes:
            for b2 in b_list:
                ix1 = max(box[0], b2[0])
                iy1 = max(box[1], b2[1])
                ix2 = min(box[2], b2[2])
                iy2 = min(box[3], b2[3])
                iw = max(ix2 - ix1, 0)
                ih = max(iy2 - iy1, 0)
                inter = iw * ih
                union = (
                    (box[2] - box[0]) * (box[3] - box[1])
                    + (b2[2] - b2[0]) * (b2[3] - b2[1])
                    - inter
                    + 1e-6
                )
                if inter / union >= iou_thr:
                    count += 1
                    break
        if count >= min_models:
            keep_b.append(box)
            keep_s.append(score)
            keep_l.append(label)
    return filter_boxes(keep_b, keep_s, keep_l)


# 시각화 함수들
def draw_boxes(frame, boxes, scores, labels):
    h, w = frame.shape[:2]
    for b, s, l in zip(boxes, scores, labels):
        x1, y1 = int(b[0] * w), int(b[1] * h)
        x2, y2 = int(b[2] * w), int(b[3] * h)
        color = (0, 0, 255) if l == 0 else (0, 255, 255)
        name = ["fire", "smoke"][l]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{name} {s:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return frame


def draw_masks(frame, masks, color=(0, 0, 255), alpha=0.4):
    if masks is None:
        return frame
    h, w = frame.shape[:2]
    for mask in masks:
        m = mask.cpu().numpy().astype(np.uint8)
        rm = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        cm = np.zeros_like(frame, dtype=np.uint8)
        cm[rm > 0.5] = color
        frame = cv2.addWeighted(cm, alpha, frame, 1 - alpha, 0)
    return frame


# 비디오 처리 함수
def process_video(cap, limit_time=60):
    global frame_count, inference_times, detection_log, alert_log, first_alert_image
    frame_count, inference_times = 0, []
    detection_log, alert_log = [], []
    first_alert_image = None
    start_time = time.time()
    first_alert_time, first_alert_reason = None, None

    while cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 360))
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=15)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t0 = time.time()
        preds = [m(frame, conf=0.3, iou=0.3, verbose=False)[0] for m in models]
        consensus = 2 if len(preds) > 1 else 1
        boxes, scores, labels = ensemble_with_consensus(preds, min_models=consensus)

        # 세그 수행 여부 판단
        seg_masks = None
        run_seg = any(
            (lbl == 1 and scr >= 0.5) or (lbl == 0 and scr >= 0.7)
            for lbl, scr in zip(labels, scores)
        )
        if run_seg:
            for name, m in zip(selected_model_names, models):
                if "seg" in name.lower():
                    res = m(frame, conf=0.3, iou=0.3, verbose=False)[0]
                    seg_masks = res.masks.data if res.masks else None
                    break
        t1 = time.time()

        # ── 추가: 연기 마스크 팽창률 계산 ──
        smoke_mask_growth = 1.0
        if seg_masks is not None:
            h_f, w_f = frame.shape[:2]
            agg = np.zeros((h_f, w_f), dtype=np.uint8)
            for m in seg_masks:
                m_np = m.cpu().numpy().astype(np.uint8)
                rm = cv2.resize(m_np, (w_f, h_f), interpolation=cv2.INTER_NEAREST)
                agg |= (rm > 0.5).astype(np.uint8)
            curr_area = int(agg.sum())
            prev_area = st.session_state.prev_smoke_mask_area
            smoke_mask_growth = curr_area / (prev_area + 1e-6)
            st.session_state.prev_smoke_mask_area = curr_area

        # 시각화
        ann = draw_masks(frame.copy(), seg_masks)
        ann = draw_boxes(ann, boxes, scores, labels)
        FRAME_WINDOW.image(ann, channels="BGR", use_container_width=True)

        now = time.strftime("%H:%M:%S", time.localtime())
        fire_area = sum(
            (b[2] - b[0]) * (b[3] - b[1]) for b, l in zip(boxes, labels) if l == 0
        )
        gr = (
            fire_area / st.session_state.prev_fire_area
            if st.session_state.prev_fire_area > 0
            else 1
        )
        st.session_state.prev_fire_area = fire_area

        for b, s, l in zip(boxes, scores, labels):
            cls = "fire" if l == 0 else "smoke"
            coord = tuple(round(x, 2) for x in b)
            key = str(coord)
            ig = 1.0
            if cls == "smoke":
                h_g, w_g = gray.shape
                x1, y1 = int(b[0] * w_g), int(b[1] * h_g)
                x2, y2 = int(b[2] * w_g), int(b[3] * h_g)
                roi = gray[y1:y2, x1:x2]
                if roi.size > 0:
                    mi = float(np.mean(roi))
                    pi = st.session_state.prev_smoke_intensity.get(key, mi)
                    ig = mi / (pi + 1e-6)
                    st.session_state.prev_smoke_intensity[key] = mi

            detection_log.append(
                {"시간": now, "클래스": cls, "신뢰도": round(s, 2), "좌표": str(coord)}
            )

            # ── 화재/연기 리스크 판정 ──
            risk, reason = False, ""
            if cls == "fire":
                if not allow_fire and s >= 0.6:
                    risk, reason = True, "허용되지 않은 위치 신뢰도≥0.6"
                elif gr > 3 and s >= 0.7:
                    risk, reason = True, "면적성장률>3 신뢰도≥0.7"
                # ★ 추가: 연기 영역 급팽창 시 화재 위험
                elif smoke_mask_growth > 1.5:
                    risk, reason = True, f"연기영역팽창>1.5 (x{smoke_mask_growth:.2f})"
            else:
                if ig > 1.2:
                    risk, reason = True, f"연기농도증가율>1.2 (x{ig:.2f})"
                elif s >= 0.7:
                    risk, reason = True, "연기신뢰도≥0.7"

            if risk:
                alert_log.append(
                    {
                        "⏰ 시간": now,
                        "⚠️ 위험": cls,
                        "신뢰도": round(s, 2),
                        "농도증가율": round(ig, 2),
                        "좌표": str(coord),
                    }
                )
                if first_alert_time is None:
                    first_alert_time = time.time() - start_time
                    first_alert_image = ann.copy()
                    first_alert_reason = reason

        detection_log, alert_log = detection_log[-100:], alert_log[-50:]
        log_table.dataframe(pd.DataFrame(detection_log), use_container_width=True)
        alert_table.dataframe(pd.DataFrame(alert_log), use_container_width=True)

        inference_times.append(t1 - t0)
        frame_count += 1
        if time.time() - start_time > limit_time:
            break

    if cap:
        cap.release()

    if frame_count > 0:
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed
        avg_inf = sum(inference_times) / len(inference_times)
        inf_fps = 1 / avg_inf if avg_inf > 0 else 0
        st.markdown("## 📊 분석 결과")
        st.write(f"🔁 총 프레임 수: {frame_count}")
        st.write(f"⏱️ 총 시간: {elapsed:.2f}s")
        st.write(f"📸 평균 FPS: {avg_fps:.2f}")
        st.write(f"🧠 평균 추론 시간: {avg_inf:.4f}s/frame")
        st.write(f"⚡ 순수 추론 FPS: {inf_fps:.2f}")
        if first_alert_time is not None:
            st.markdown(f"🚨 최초 경고 시점: **{first_alert_time:.2f}초**")
            st.markdown(f"**최초 경고 이유:** {first_alert_reason}")
            st.image(
                cv2.cvtColor(first_alert_image, cv2.COLOR_BGR2RGB),
                caption="🚨 최초 경고 이미지",
                use_container_width=True,
            )


# 실행
if cap:
    process_video(cap)
