"""
A-5: 경보 알림 이메일 발송 (smtplib)
- ALERT_EMAIL: 수신인 (.env)
- SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASS: 발신 서버 설정 (.env)
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional


def _cfg(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def send_alert(
    subject: str,
    body: str,
    to: Optional[str] = None,
) -> bool:
    """
    이메일 발송. 성공 시 True, 실패(설정 미비 포함) 시 False.
    """
    recipient = to or _cfg("ALERT_EMAIL")
    smtp_host = _cfg("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(_cfg("SMTP_PORT", "587"))
    smtp_user = _cfg("SMTP_USER")
    smtp_pass = _cfg("SMTP_PASS")

    if not recipient or not smtp_user or not smtp_pass:
        print("[Notifier] ALERT_EMAIL / SMTP 설정이 없어 이메일 발송 건너뜀")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = smtp_user
        msg["To"]      = recipient
        msg.attach(MIMEText(body, "plain", "utf-8"))

        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, recipient, msg.as_string())

        print(f"[Notifier] 경보 이메일 발송 완료 → {recipient}")
        return True

    except Exception as e:
        print(f"[Notifier] 이메일 발송 실패: {e}")
        return False


def send_trigger_alert(
    event_type: str,
    reason: str,
    predictions: dict,
    timestamp: Optional[str] = None,
) -> bool:
    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M")
    solar = predictions.get("solar_mw", 0.0)
    wind  = predictions.get("wind_mw",  0.0)
    subject = f"[제주 에너지 에이전트] {event_type} 경보 — {ts}"
    body = (
        f"■ 경보 유형: {event_type}\n"
        f"■ 발생 시각: {ts}\n"
        f"■ 원인: {reason}\n\n"
        f"■ 예측 결과\n"
        f"  태양광: {solar:.2f} MW\n"
        f"  풍  력: {wind:.2f} MW\n"
        f"  합  계: {solar + wind:.2f} MW\n\n"
        "본 메일은 자동 발송입니다."
    )
    return send_alert(subject, body)
