import streamlit as st
import cv2
import time
import tempfile
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from collections import defaultdict

# ─── 페이지 레이아웃 설정 ─────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🔥 화재/연기 실시간 탐지 시스템")

# ─── 모델 옵션 정의 ─────────────────────────────────────────────────
model_options = {
    "YOLO11n-seg-smoke": "runs/segment/firesmoke_seg_v11_smoke_focus/weights/best.pt",
    "YOLOv8s (최고 성능)": "runs/detect/firesmoke_detect_v8s/weights/best.pt",
}

# ─── 모델 로드 ─────────────────────────────────────────────────────
v8s_model = YOLO(model_options["YOLOv8s (최고 성능)"])
seg_model = YOLO(model_options["YOLO11n-seg-smoke"])

# ─── 플레이스홀더 & 설정 ───────────────────────────────────────────
FRAME_WINDOW = st.empty()
TEST_DURATION = 60  # 테스트 지속 시간(초)

# ─── 전역/세션 상태 초기화 ─────────────────────────────────────────
frame_count = 0
inference_times = []
detection_log = []
alert_log = []
first_alert_time = None
first_alert_reason = None
first_alert_image = None

if "prev_fire_area" not in st.session_state:
    st.session_state.prev_fire_area = 0
if "prev_smoke_mask_area" not in st.session_state:
    st.session_state.prev_smoke_mask_area = 0
if "prev_smoke_intensity" not in st.session_state:
    st.session_state.prev_smoke_intensity = {}

# 누적 알림 카운터
if "warning_count" not in st.session_state:
    st.session_state.warning_count = 0
if "danger_count" not in st.session_state:
    st.session_state.danger_count = 0

# 한 번만 띄울 알림 플래그
if "initial_alerted" not in st.session_state:
    st.session_state.initial_alerted = False
if "threshold_warning_alerted" not in st.session_state:
    st.session_state.threshold_warning_alerted = False
if "threshold_danger_alerted" not in st.session_state:
    st.session_state.threshold_danger_alerted = False

# ─── 카메라 위치 설정 ───────────────────────────────────────────────
LOCATION_INFO = {
    "cam01": {"type": "주방", "allow_fire": True},
    "cam02": {"type": "거실", "allow_fire": False},
    "cam03": {"type": "창고", "allow_fire": False},
    "cam04": {"type": "작업실", "allow_fire": True},
}
display_options = {f"{v['type']} ({k})": k for k, v in LOCATION_INFO.items()}
selected_display = st.selectbox("📍 카메라 위치 선택", list(display_options.keys()))
cam_id = display_options[selected_display]
allow_fire = LOCATION_INFO[cam_id]["allow_fire"]

# ─── 로그 탭 ───────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📋 전체 탐지 로그", "🚨 경고 대시보드"])
log_table = tab1.empty()
alert_table = tab2.empty()


# ─── 유틸 함수들 ────────────────────────────────────────────────────
def filter_boxes(boxes, scores, labels, min_area=0.001, max_area=0.4, conf_thr=0.3):
    out_b, out_s, out_l = [], [], []
    for b, s, l in zip(boxes, scores, labels):
        if s < conf_thr:
            continue
        w, h = b[2] - b[0], b[3] - b[1]
        area = w * h
        if min_area <= area <= max_area:
            out_b.append(b)
            out_s.append(s)
            out_l.append(int(l))
    return out_b, out_s, out_l


def ensemble_predictions(preds, iou_thr=0.5, skip_thr=0.001):
    all_b, all_s, all_l = [], [], []
    for p in preds:
        h, w = p.orig_shape
        b_list, s_list, l_list = [], [], []
        for box, score, lab in zip(
            p.boxes.xyxy.cpu().numpy(),
            p.boxes.conf.cpu().numpy(),
            p.boxes.cls.cpu().numpy(),
        ):
            x1, y1, x2, y2 = box
            b_list.append([x1 / w, y1 / h, x2 / w, y2 / h])
            s_list.append(float(score))
            l_list.append(int(lab))
        all_b.append(b_list)
        all_s.append(s_list)
        all_l.append(l_list)
    fb, fs, fl = weighted_boxes_fusion(
        all_b, all_s, all_l, iou_thr=iou_thr, skip_box_thr=skip_thr
    )
    return filter_boxes(fb, fs, fl)


def draw_boxes(frame, boxes, scores, labels):
    h, w = frame.shape[:2]
    for b, s, l in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [int(v * d) for v, d in zip(b, (w, h, w, h))]
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
    if not isinstance(masks, (list, tuple)) or len(masks) == 0:
        return frame
    h, w = frame.shape[:2]
    for m in masks:
        m_np = m.cpu().numpy().astype(np.uint8)
        rm = cv2.resize(m_np, (w, h), interpolation=cv2.INTER_NEAREST)
        cm = np.zeros_like(frame, dtype=np.uint8)
        cm[rm > 0.5] = color
        frame = cv2.addWeighted(cm, alpha, frame, 1 - alpha, 0)
    return frame


# ─── 메인 처리 루프 ─────────────────────────────────────────────────
def process_video(cap, limit_time=60):
    global frame_count, inference_times, detection_log, alert_log
    global first_alert_time, first_alert_reason, first_alert_image

    frame_count = 0
    inference_times = []
    detection_log = []
    alert_log = []
    start_time = time.time()
    first_alert_time = None
    first_alert_reason = None
    first_alert_image = None
    st.session_state.initial_alerted = False  # reset at start

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 전처리
        frame = cv2.resize(frame, (640, 360))
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=15)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        # 추론
        t0 = time.time()
        pred = v8s_model(frame, conf=0.3, iou=0.3, verbose=False)[0]
        boxes, scores, labels = ensemble_predictions([pred])

        # smoke mask 성장률 계산
        seg_masks = []
        smoke_growth = 1.0
        if 1 in labels:
            res = seg_model(frame, conf=0.3, iou=0.3, verbose=False)[0]
            if res.masks is not None and len(res.masks.data) > 0:
                seg_masks = list(res.masks.data)
                agg = np.zeros((h, w), dtype=np.uint8)
                for m in seg_masks:
                    m_np = m.cpu().numpy().astype(np.uint8)
                    rm = cv2.resize(m_np, (w, h), interpolation=cv2.INTER_NEAREST)
                    agg |= (rm > 0.5).astype(np.uint8)
                curr_area = int(agg.sum())
                prev_area = st.session_state.prev_smoke_mask_area
                smoke_growth = curr_area / (prev_area + 1e-6)
                st.session_state.prev_smoke_mask_area = curr_area

        inference_times.append(time.time() - t0)

        # 시각화
        vis = draw_masks(frame.copy(), seg_masks)
        vis = draw_boxes(vis, boxes, scores, labels)
        FRAME_WINDOW.image(vis, channels="BGR", use_container_width=True)

        # 시간 & 면적 변화
        now = time.strftime("%H:%M:%S")
        fire_area = sum(
            (b[2] - b[0]) * (b[3] - b[1]) for b, l in zip(boxes, labels) if l == 0
        )
        growth_ratio = fire_area / (st.session_state.prev_fire_area + 1e-6)
        st.session_state.prev_fire_area = fire_area

        # 프레임별 처리
        for b, s, l in zip(boxes, scores, labels):
            cls = "fire" if l == 0 else "smoke"
            coord = tuple(round(x, 2) for x in b)

            # detection 로그
            detection_log.append(
                {"시간": now, "클래스": cls, "신뢰도": round(s, 2), "좌표": coord}
            )

            # smoke 농도 변화율
            ig = 1.0
            if cls == "smoke":
                x1, y1, x2, y2 = [int(v * d) for v, d in zip(b, (w, h, w, h))]
                roi = gray[y1:y2, x1:x2]
                if roi.size > 0:
                    mi = float(np.mean(roi))
                    prev_int = st.session_state.prev_smoke_intensity.get(str(coord), mi)
                    ig = mi / (prev_int + 1e-6)
                    st.session_state.prev_smoke_intensity[str(coord)] = mi

            # 리스크 판단 & 레벨 직접 설정
            risk = False
            reason = ""
            level = "warning"  # 기본 warning

            if cls == "fire":
                if not allow_fire and s >= 0.6:
                    risk, reason = True, "허용되지 않은 위치(신뢰도>=0.6)"
                elif growth_ratio > 3 and s >= 0.7:
                    risk, reason = True, f"화재 면적 급성장 ({growth_ratio:.1f}배)"
                    level = "danger"
                elif smoke_growth > 1.5:
                    risk, reason = True, f"연기영역팽창>1.5x({smoke_growth:.2f})"
            else:  # smoke
                if not allow_fire:
                    risk, reason = True, "허용되지 않은 위치에서 연기 감지됨"
                elif s >= 0.7 and ig > 1.1:
                    risk, reason = True, f"연기신뢰도≥0.7 & 농도증가율>1.1x({ig:.2f})"
                    level = "caution"
                elif ig > 1.5:
                    risk, reason = True, f"연기농도증가율>1.5x({ig:.2f})"
                    level = "danger"
                elif smoke_growth > 1.3:
                    risk, reason = True, f"연기영역팽창>1.3x({smoke_growth:.2f})"

            if risk:
                severity_text = level  # "caution"/"warning"/"danger"

                # 최초 alert (warning or danger) 1회
                if not st.session_state.initial_alerted and severity_text in (
                    "warning",
                    "danger",
                ):
                    st.warning(f"🚨 최초 경고 ({now}): {reason}")
                    first_alert_time = time.time() - start_time
                    first_alert_reason = reason
                    first_alert_image = vis.copy()
                    st.session_state.initial_alerted = True

                # 누적 카운트
                if severity_text == "warning":
                    st.session_state.warning_count += 1
                elif severity_text == "danger":
                    st.session_state.danger_count += 1

                # alert 로그
                alert_log.append(
                    {
                        "시간": now,
                        "위험": cls,
                        "신뢰도": round(s, 2),
                        "농도증가율": round(ig, 2) if cls == "smoke" else "-",
                        "좌표": coord,
                        "원인": reason,
                        "레벨": severity_text,
                    }
                )

        # 테이블 갱신
        log_table.dataframe(pd.DataFrame(detection_log), use_container_width=True)
        alert_table.dataframe(pd.DataFrame(alert_log), use_container_width=True)

        # 누적 warning 10회 → 경고 알림 1회
        if (
            st.session_state.warning_count == 10
            and not st.session_state.threshold_warning_alerted
        ):
            st.warning("⚠️ warning 10회 누적: 경고 알림")
            st.session_state.threshold_warning_alerted = True

        # 누적 danger 5회 → 위험 알림 1회
        if (
            st.session_state.danger_count == 5
            and not st.session_state.threshold_danger_alerted
        ):
            st.error("🛑 danger 5회 누적: 위험 알림")
            st.session_state.threshold_danger_alerted = True

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


# ─── 입력 소스별 처리 ─────────────────────────────────────────────
option = st.radio("🎥 입력 속성 선택", ["웹캠", "영상 업로드"])
if option == "웹캠":
    if st.checkbox("▶️ 웹캠 시작"):
        cap = cv2.VideoCapture(0)
        elapsed, fa_t, fa_img, fa_r = process_video(cap, TEST_DURATION)
elif option == "영상 업로드":
    file = st.file_uploader("📁 영상 업로드", type=["mp4", "avi", "mov"])
    if file:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(file.read())
        cap = cv2.VideoCapture(tmp.name)
        elapsed, fa_t, fa_img, fa_r = process_video(cap, TEST_DURATION)

# ─── 결과 출력 ───────────────────────────────────────────────────
if frame_count > 0:
    avg_fps = frame_count / elapsed
    avg_inf = sum(inference_times) / len(inference_times)
    st.markdown("## 📊 분석 결과")
    st.write(f"🔁 총 프레임 수: {frame_count}")
    st.write(f"⏱️ 총 시간: {elapsed:.2f}s")
    st.write(f"📸 평균 FPS: {avg_fps:.2f}")
    st.write(f"🧠 평균 추론 시간: {avg_inf:.4f}s/frame")
    if first_alert_time is not None:
        st.markdown(f"🚨 최초 경고 시점: **{first_alert_time:.2f}초**")
        st.markdown(f"**최초 경고 원인:** {first_alert_reason}")
        st.image(
            cv2.cvtColor(first_alert_image, cv2.COLOR_BGR2RGB),
            caption="🚨 최초 경고 이미지",
            use_container_width=True,
        )
