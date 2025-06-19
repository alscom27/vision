# app.py

import streamlit as st
import cv2
import time
import tempfile
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
import threading
import email_utils  # send_alert_email_with_image 포함

from utils.detection_utils import (
    filter_boxes,
    ensemble_predictions,
    draw_boxes,
    draw_masks,
)
from utils.layout import setup_layout, display_camera_selector, display_tabs
from logic.alert_logic import (
    reset_alert_state,
    check_first_alert,
    check_threshold_alerts,
    evaluate_risks,
    evaluate_risks_from_masks,
)

st.set_page_config(layout="wide")
setup_layout()

model_options = {
    "YOLO11n-seg-smoke": "runs/segment/firesmoke_seg_v11_smoke_focus/weights/best.pt",
    "YOLOv8s (최고 성능)": "runs/detect/firesmoke_detect_v8s/weights/best.pt",
}
v8s_model = YOLO(model_options["YOLOv8s (최고 성능)"])
seg_model = YOLO(model_options["YOLO11n-seg-smoke"])

FRAME_WINDOW = st.empty()
TEST_DURATION = 60
if "prev_fire_area" not in st.session_state:
    reset_alert_state()

frame_count = 0
inference_times = []
detection_log = []
alert_log = []
first_alert_time, first_alert_reason, first_alert_image = None, None, None

cam_id, allow_fire, selected_display = display_camera_selector()
log_table, alert_table = display_tabs()


def process_video(cap, limit_time=60):
    global frame_count, inference_times, detection_log, alert_log
    global first_alert_time, first_alert_reason, first_alert_image

    frame_count = 0
    inference_times.clear()
    detection_log.clear()
    alert_log.clear()
    first_alert_time, first_alert_reason, first_alert_image = None, None, None
    st.session_state.initial_alerted = False

    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 360))
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=15)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        t0 = time.time()
        pred = v8s_model(frame, conf=0.3, iou=0.3, verbose=False)[0]
        boxes, scores, labels = ensemble_predictions([pred])
        inference_times.append(time.time() - t0)

        now = time.strftime("%H:%M:%S")
        seg_masks, smoke_growth = [], 1.0
        if 1 in labels:
            res = seg_model(frame, conf=0.3, iou=0.3, verbose=False)[0]
            if res.masks and len(res.masks.data) > 0:
                seg_masks, smoke_growth = evaluate_risks_from_masks(res, h, w)

        vis = draw_masks(frame.copy(), seg_masks)
        vis = draw_boxes(vis, boxes, scores, labels)
        FRAME_WINDOW.image(vis, channels="BGR", use_container_width=True)

        now_log, alert_result, fa_img, fa_reason, fa_time = check_first_alert(
            boxes, scores, labels, vis, selected_display, start, now
        )
        detection_log.extend(now_log)
        alert_log.extend(alert_result)

        if fa_img and not first_alert_image:
            first_alert_image = fa_img
            first_alert_reason = fa_reason
            first_alert_time = fa_time

        alert_result2, fa_img2, fa_reason2, fa_time2 = evaluate_risks(
            boxes,
            scores,
            labels,
            gray,
            frame.shape,
            vis,
            smoke_growth,
            cam_id,
            allow_fire,
            selected_display,
            start,
            now,
        )
        alert_log.extend(alert_result2)
        if fa_img2 is not None and not first_alert_image:
            first_alert_image = fa_img2
            first_alert_reason = fa_reason2
            first_alert_time = fa_time2

        log_table.dataframe(pd.DataFrame(detection_log), use_container_width=True)
        alert_table.dataframe(pd.DataFrame(alert_log), use_container_width=True)

        check_threshold_alerts(vis, selected_display)

        frame_count += 1
        if time.time() - start > limit_time:
            break

    cap.release()
    return time.time() - start, first_alert_time, first_alert_image, first_alert_reason


option = st.radio("🎥 입력 속성 선택", ["웹캠", "영상 업로드"])
if option == "웹캠":
    if st.checkbox("▶️ 웹캠 시작"):
        reset_alert_state()
        cap = cv2.VideoCapture(0)
        elapsed, fa_t, fa_img, fa_r = process_video(cap, TEST_DURATION)
else:
    file = st.file_uploader("📁 영상 업로드", type=["mp4", "avi", "mov"])
    if file:
        reset_alert_state()
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(file.read())
        cap = cv2.VideoCapture(tmp.name)
        elapsed, fa_t, fa_img, fa_r = process_video(cap, TEST_DURATION)

if frame_count > 0:
    avg_fps = frame_count / elapsed
    avg_inf = (sum(inference_times) / len(inference_times)) if inference_times else 0
    st.markdown("## 📊 분석 결과")
    st.write(f"🔁 총 프레임 수: {frame_count}")
    st.write(f"⏱️ 총 시간: {elapsed:.2f}s")
    st.write(f"📸 평균 FPS: {avg_fps:.2f}")
    st.write(f"🧠 평균 추론 시간: {avg_inf:.4f}s/frame")
    if fa_t is not None:
        st.markdown(f"🚨 최초 경고 시점: **{fa_t:.2f}초**")
        st.markdown(f"**최초 경고 원인:** {fa_r}")
        st.image(
            cv2.cvtColor(fa_img, cv2.COLOR_BGR2RGB),
            caption="🚨 최초 경고 이미지",
            use_container_width=True,
        )
