# alert_logic.py

import cv2
import streamlit as st
import time
import numpy as np
import email_utils
from threading import Thread


def reset_alert_state():
    st.session_state.initial_alerted = False
    st.session_state.threshold_warning_alerted = False
    st.session_state.threshold_danger_alerted = False
    st.session_state.warning_count = 0
    st.session_state.danger_count = 0
    st.session_state.prev_fire_area = 0
    st.session_state.prev_smoke_mask_area = 0
    st.session_state.prev_smoke_intensity = {}


def evaluate_risks_from_masks(result, h, w):
    seg_masks = list(result.masks.data)
    agg = np.zeros((h, w), dtype=np.uint8)
    for m in seg_masks:
        m_np = m.cpu().numpy().astype(np.uint8)
        rm = cv2.resize(m_np, (w, h), interpolation=cv2.INTER_NEAREST)
        agg |= (rm > 0.5).astype(np.uint8)
    curr = agg.sum()
    prev = st.session_state.prev_smoke_mask_area
    growth = curr / (prev + 1e-6)
    st.session_state.prev_smoke_mask_area = curr
    return seg_masks, growth


def check_first_alert(boxes, scores, labels, vis, selected_display, start, now):
    log = []
    alert = []
    fa_img, fa_reason, fa_time = None, None, None

    if labels.count(1) >= 3 and not st.session_state.initial_alerted:
        reason = "연기 객체 3개 이상"
        st.warning(f"🚨 최초 경고 ({now}): {reason}")
        annotated = vis.copy()

        def send():
            email_utils.send_alert_email_with_image(
                f"🚨 최초 경고 발생: {selected_display}",
                f"시간: {now}\n위치: {selected_display}\n원인: {reason}",
                annotated,
            )

        Thread(target=send, daemon=True).start()

        st.session_state.initial_alerted = True
        fa_img, fa_reason, fa_time = annotated, reason, time.time() - start
        alert.append(
            {
                "시간": now,
                "위험": "smoke",
                "신뢰도": "-",
                "농도증가율": "-",
                "좌표": "-",
                "원인": reason,
                "레벨": "warning",
            }
        )

    for b, s, l in zip(boxes, scores, labels):
        name = "fire" if l == 0 else "smoke"
        coord = tuple(round(x, 2) for x in b)
        log.append(
            {
                "시간": now,
                "클래스": name,
                "신뢰도": round(s, 2),
                "좌표": coord,
            }
        )

    return log, alert, fa_img, fa_reason, fa_time


def check_threshold_alerts(vis, selected_display):
    def send(title, msg):
        email_utils.send_alert_email_with_image(title, msg, vis)

    if (
        st.session_state.warning_count == 10
        and not st.session_state.threshold_warning_alerted
    ):
        st.warning("⚠️ warning 10회 누적: 경고 알림")
        Thread(
            target=send,
            args=(
                f"⚠️ Warning 10회 누적: {selected_display}",
                f"현재까지 warning이 10회 누적되었습니다.\n위치: {selected_display}",
            ),
            daemon=True,
        ).start()
        st.session_state.threshold_warning_alerted = True

    if (
        st.session_state.danger_count == 5
        and not st.session_state.threshold_danger_alerted
    ):
        st.error("🛑 danger 5회 누적: 위험 알림")
        Thread(
            target=send,
            args=(
                f"🛑 Danger 5회 누적: {selected_display}",
                f"현재까지 danger가 5회 누적되었습니다.\n위치: {selected_display}",
            ),
            daemon=True,
        ).start()
        st.session_state.threshold_danger_alerted = True


def evaluate_risks(
    boxes,
    scores,
    labels,
    gray,
    shape,
    vis,
    smoke_growth,
    cam_id,
    allow_fire,
    selected_display,
    start,
    now,
):
    h, w = shape[:2]
    fire_area = sum(
        (b[2] - b[0]) * (b[3] - b[1]) for b, l in zip(boxes, labels) if l == 0
    )
    growth = fire_area / (st.session_state.prev_fire_area + 1e-6)
    st.session_state.prev_fire_area = fire_area

    alert_result = []
    fa_img, fa_reason, fa_time = None, None, None

    for b, s, l in zip(boxes, scores, labels):
        cls = "fire" if l == 0 else "smoke"
        coord = tuple(round(x, 2) for x in b)
        ig = 1.0
        if cls == "smoke":
            x1, y1, x2, y2 = [int(v * d) for v, d in zip(b, (w, h, w, h))]
            roi = gray[y1:y2, x1:x2]
            if roi.size > 0:
                mi = float(np.mean(roi))
                prev = st.session_state.prev_smoke_intensity.get(str(coord), mi)
                ig = mi / (prev + 1e-6)
                st.session_state.prev_smoke_intensity[str(coord)] = mi

        risk = False
        level = "warning"
        reason = ""

        if cls == "fire":
            if not allow_fire and s >= 0.6:
                risk, reason = True, "허용되지 않은 위치 불 감지됨 (신뢰도>=0.6)"
            elif growth > 3 and s >= 0.7:
                risk, level, reason = (
                    True,
                    "danger",
                    f"화재 면적 급성장({growth:.1f}배)",
                )
            elif smoke_growth > 1.5:
                risk, reason = True, f"연기영역팽창>1.5x({smoke_growth:.2f})"
        else:
            if not allow_fire:
                risk, reason = True, "허용되지 않은 위치에서 연기 감지됨"
            elif s >= 0.7 and ig > 1.1:
                risk, level, reason = (
                    True,
                    "caution",
                    f"연기신뢰도≥0.7 & 농도증가율>1.1x({ig:.2f})",
                )
            elif ig > 1.5:
                risk, level, reason = True, "danger", f"연기농도증가율>1.5x({ig:.2f})"
            elif smoke_growth > 1.3:
                risk, reason = True, f"연기영역팽창>1.3x({smoke_growth:.2f})"

        if risk:
            if not st.session_state.initial_alerted:
                st.warning(f"🚨 최초 경고 ({now}): {reason}")
                Thread(
                    target=email_utils.send_alert_email_with_image,
                    args=(
                        f"🚨 최초 경고 발생: {selected_display}",
                        f"시간: {now}\n위치: {selected_display}\n원인: {reason}",
                        vis.copy(),
                    ),
                    daemon=True,
                ).start()
                st.session_state.initial_alerted = True
                fa_img, fa_reason, fa_time = vis.copy(), reason, time.time() - start

            if level == "warning":
                st.session_state.warning_count += 1
            if level == "danger":
                st.session_state.danger_count += 1

            alert_result.append(
                {
                    "시간": now,
                    "위험": cls,
                    "신뢰도": round(s, 2),
                    "농도증가율": round(ig, 2) if cls == "smoke" else "-",
                    "좌표": coord,
                    "원인": reason,
                    "레벨": level,
                }
            )

    return alert_result, fa_img, fa_reason, fa_time
