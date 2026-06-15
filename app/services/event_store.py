"""
A-3: 트리거·경보 이벤트 영속화
- data/events.json 에 append 방식으로 저장
- get_events() / append_event() 공개 인터페이스
"""
import json
import os
from datetime import datetime
from typing import Any, Optional

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVENTS_PATH = os.path.join(BASE_DIR, "data", "events.json")


def _ensure_file():
    os.makedirs(os.path.dirname(EVENTS_PATH), exist_ok=True)
    if not os.path.exists(EVENTS_PATH):
        with open(EVENTS_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)


def append_event(
    event_type: str,
    reason: str,
    action: str,
    prediction_after: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> dict[str, Any]:
    """이벤트를 파일에 추가하고 저장된 이벤트 반환"""
    _ensure_file()
    event: dict[str, Any] = {
        "timestamp":        datetime.now().isoformat(timespec="seconds"),
        "type":             event_type,
        "reason":           reason,
        "action":           action,
        "prediction_after": prediction_after or {},
    }
    if extra:
        event.update(extra)

    with open(EVENTS_PATH, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data.append(event)
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.truncate()

    return event


def get_events(
    limit: int = 50,
    event_type: Optional[str] = None,
) -> list[dict]:
    """최신순으로 이벤트 목록 반환"""
    _ensure_file()
    try:
        with open(EVENTS_PATH, "r", encoding="utf-8") as f:
            data: list[dict] = json.load(f)
    except json.JSONDecodeError:
        return []

    if event_type:
        data = [e for e in data if e.get("type") == event_type]

    return list(reversed(data))[:limit]
