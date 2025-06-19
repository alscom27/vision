# layout.py

import streamlit as st

# ─── 카메라 위치 설정 ─────────────────────────────────────
LOCATION_INFO = {
    "cam01": {"type": "주방", "allow_fire": True},
    "cam02": {"type": "거실", "allow_fire": False},
    "cam03": {"type": "창고", "allow_fire": False},
    "cam04": {"type": "작업실", "allow_fire": True},
}


def setup_layout():
    st.title("🔥 화재 경고 알림 서비스 데모")
    st.markdown("### 화재/연기 실시간 탐지 시스템")
    st.markdown("- 사용 모델 : YOLOv8s, YOLO11n-seg")
    st.markdown("---")


def display_camera_selector():
    display_options = {f"{v['type']} ({k})": k for k, v in LOCATION_INFO.items()}
    selected_display = st.selectbox("📍 카메라 위치 선택", list(display_options.keys()))
    cam_id = display_options[selected_display]
    allow_fire = LOCATION_INFO[cam_id]["allow_fire"]
    return cam_id, allow_fire, selected_display


def display_tabs():
    tab1, tab2 = st.tabs(["📋 전체 탐지 로그", "🚨 경고 대시보드"])
    log_table = tab1.empty()
    alert_table = tab2.empty()
    return log_table, alert_table
