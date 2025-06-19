# email_utils.py

import os
import cv2
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from email.mime.image import MIMEImage

# .env 로드
load_dotenv()

# 환경변수에서 이메일 정보 가져오기
EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_TO = os.getenv("EMAIL_TO")


def send_alert_email_with_image(subject, body, image, to_email=EMAIL_TO):
    """
    Gmail SMTP를 통해 경고 이메일을 전송합니다.

    Parameters:
        subject (str): 이메일 제목
        body (str): 이메일 본문
        to_email (str): 수신자 이메일 (기본값: .env에 지정된 EMAIL_TO)
    """
    msg = MIMEMultipart()
    msg["From"] = EMAIL_FROM
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # 이미지 첨부 (OpenCV 이미지 → JPEG 변환)
    _, img_encoded = cv2.imencode(".jpg", image)
    image_data = img_encoded.tobytes()
    image_part = MIMEImage(image_data, name="alert.jpg")
    msg.attach(image_part)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, to_email, msg.as_string())
        print(f"✅ 메일 전송 성공 (이미지 포함)")
    except Exception as e:
        print(f"❌ 메일 전송 실패: {e}")
