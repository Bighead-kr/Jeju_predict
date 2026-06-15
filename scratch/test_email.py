import os
import sys
from dotenv import load_dotenv

# 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(".env")

from app.services import notifier

print("ALERT_EMAIL:", os.getenv("ALERT_EMAIL"))
print("SMTP_HOST:", os.getenv("SMTP_HOST"))
print("SMTP_PORT:", os.getenv("SMTP_PORT"))
print("SMTP_USER:", os.getenv("SMTP_USER"))
print("SMTP_PASS:", os.getenv("SMTP_PASS"))

predictions = {"solar_mw": 45.2, "wind_mw": 37.1}
weather_info = {"temp": 20.1, "wind_speed": 4.5, "solar_rad": 1.5}

try:
    print("Sending briefing email...")
    res = notifier.send_briefing_email(
        briefing_type="일출 전 예측 브리핑",
        predictions=predictions,
        weather_info=weather_info
    )
    print("Result:", res)
except Exception as e:
    import traceback
    traceback.print_exc()
