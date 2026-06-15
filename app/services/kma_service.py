"""
기상청 단기예보 API 클라이언트
- 제주도 격자(nx=53, ny=38) 기준 1시간 간격 예보 조회
- 9개 모델 입력 변수로 매핑하여 (24, 9) numpy array 반환
"""
import os
import math
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

API_URL = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
JEJU_NX = 53
JEJU_NY = 38
# 단기예보 발표시각 (매일 8회)
BASE_TIMES = ["0200", "0500", "0800", "1100", "1400", "1700", "2000", "2300"]

SKY_TO_CLOUD = {1: 2.0, 2: 4.0, 3: 6.5, 4: 9.0}  # 1:맑음 2:구름조금 3:구름많음 4:흐림


def _latest_base_time(dt: datetime) -> tuple[str, str]:
    """dt 이전 가장 최근 발표시각(base_date, base_time) 반환"""
    dt_min = dt - timedelta(minutes=10)  # 발표 10분 후 API 반영
    for bt in reversed(BASE_TIMES):
        hour, minute = int(bt[:2]), int(bt[2:])
        candidate = dt_min.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= dt_min:
            return dt_min.strftime("%Y%m%d"), bt
    # 자정 직후면 전날 2300 사용
    prev = dt_min - timedelta(days=1)
    return prev.strftime("%Y%m%d"), "2300"


def _sunshine(hour: int, cloud: float) -> float:
    """시각과 운량으로 일사 추정 (0-1)"""
    if not (6 <= hour <= 18):
        return 0.0
    solar_angle = math.sin(math.pi * (hour - 6) / 12)
    clear_ratio = max(0.0, (10.0 - cloud) / 10.0)
    return round(solar_angle * clear_ratio, 4)


class KMAService:
    def __init__(self):
        self.api_key = os.getenv("KMA_API_KEY", "")

    def _fetch_raw(self, base_date: str, base_time: str) -> list[dict]:
        if not self.api_key:
            print("[KMA] API key 미설정 — .env의 KMA_API_KEY를 확인하세요")
            return []
        params = {
            "serviceKey": self.api_key,
            "pageNo": 1,
            "numOfRows": 1000,
            "dataType": "JSON",
            "base_date": base_date,
            "base_time": base_time,
            "nx": JEJU_NX,
            "ny": JEJU_NY,
        }
        try:
            resp = requests.get(API_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            header = data["response"]["header"]
            if header["resultCode"] != "00":
                print(f"[KMA] API 오류 ({base_date} {base_time}): {header['resultCode']} {header['resultMsg']}")
                return []
            body = data["response"]["body"]
            items = body["items"]["item"]
            return items
        except Exception as e:
            print(f"[KMA] API 호출 실패 ({base_date} {base_time}): {type(e).__name__}: {e}")
            return []

    def _parse_items(self, items: list[dict]) -> dict[datetime, dict]:
        """item 목록 → {datetime: {category: value}} 매핑"""
        result: dict[datetime, dict] = {}
        for it in items:
            try:
                dt = datetime.strptime(f"{it['fcstDate']} {it['fcstTime']}", "%Y%m%d %H%M")
                result.setdefault(dt, {})[it["category"]] = it["fcstValue"]
            except Exception:
                continue
        return result

    def build_weather_window(self, timestamp: str) -> Optional[np.ndarray]:
        """
        timestamp 기준 직전 24시간의 (24, 9) 날씨 윈도우 반환.
        API 실패 또는 데이터 부족 시 None 반환.
        """
        try:
            dt_target = datetime.strptime(timestamp[:19], "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

        # 24h 윈도우: [dt_target - 23h, dt_target]
        dt_start = dt_target - timedelta(hours=23)

        # 필요한 base time 결정: dt_start 기준
        base_date, base_time = _latest_base_time(dt_start)
        items = self._fetch_raw(base_date, base_time)

        # 데이터 부족 시 한 단계 앞 base time 추가 조회
        if len(items) < 24:
            bt_idx = BASE_TIMES.index(base_time) if base_time in BASE_TIMES else 0
            if bt_idx > 0:
                prev_bt = BASE_TIMES[bt_idx - 1]
                items += self._fetch_raw(base_date, prev_bt)

        parsed = self._parse_items(items)
        if not parsed:
            return None

        rows = []
        for i in range(24):
            slot = dt_start + timedelta(hours=i)
            slot_norm = slot.replace(minute=0, second=0, microsecond=0)
            d = parsed.get(slot_norm, {})

            try:
                wsd = float(d.get("WSD", 5.0))
                vec = float(d.get("VEC", 180.0))
                tmp = float(d.get("TMP", 18.0))
                reh = float(d.get("REH", 65.0))
                sky = int(float(d.get("SKY", 3)))
                cloud = SKY_TO_CLOUD.get(sky, 5.0)
                rad = math.radians(vec)
                sun = _sunshine(slot.hour, cloud)
                insol = sun * 2.8

                rows.append([
                    wsd,
                    math.sin(rad),
                    math.cos(rad),
                    tmp,
                    reh,
                    1012.0,  # 기압: 단기예보 미제공 → 제주 표준값
                    cloud,
                    sun,
                    insol,
                ])
            except Exception:
                return None  # 파싱 실패 시 전체 폴백

        if len(rows) != 24:
            return None

        return np.array(rows, dtype=np.float32)
