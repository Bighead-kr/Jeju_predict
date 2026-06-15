"""
A-4: APScheduler 기반 정시 자동 예측 + 능동 경보
- 06:00 / 12:00 / 18:00 제주 발전량 예측
- 트리거 조건 감지 시 event_store 기록 + notifier 호출
"""
import os
from datetime import datetime
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.services import event_store, notifier

_scheduler: Optional[AsyncIOScheduler] = None
_agent_ref = None   # JejuPredictAgent 참조 (startup 시 주입)
_chat_ref  = None   # AgenticChatService 참조


def init(agent, chat_service):
    global _agent_ref, _chat_ref
    _agent_ref = agent
    _chat_ref  = chat_service


async def _run_scheduled_prediction():
    if _agent_ref is None or _chat_ref is None:
        print("[Monitor] 에이전트 미초기화 — 예약 예측 건너뜀")
        return

    now = datetime.now()
    ts  = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[Monitor] 정시 예측 실행: {ts}")

    try:
        window = _chat_ref._build_window(ts)
        report = _agent_ref.run_pipeline_with_refinement(window, timestamp=ts)
    except Exception as e:
        print(f"[Monitor] 예측 실패: {e}")
        return

    preds = report.get("predictions", {})

    # 트리거 감지
    if report.get("trigger_active"):
        warnings = report.get("warnings", [])
        reason   = "; ".join(warnings) if warnings else "트리거 조건 충족"
        event_store.append_event(
            event_type="trigger",
            reason=reason,
            action="scheduled_prediction",
            prediction_after=preds,
        )
        notifier.send_trigger_alert(
            event_type="trigger",
            reason=reason,
            predictions=preds,
            timestamp=ts,
        )
        print(f"[Monitor] 트리거 경보 발동: {reason}")
    else:
        # 정상 예측도 이벤트로 기록 (type=info)
        event_store.append_event(
            event_type="info",
            reason="정시 예측 완료",
            action="scheduled_prediction",
            prediction_after=preds,
        )

    print(f"[Monitor] 예측 완료 — 태양광 {preds.get('solar_mw', 0):.2f} MW / "
          f"풍력 {preds.get('wind_mw', 0):.2f} MW")


def start():
    global _scheduler
    if _scheduler and _scheduler.running:
        return

    _scheduler = AsyncIOScheduler(timezone="Asia/Seoul")
    # 06:00 / 12:00 / 18:00 정시 예측
    for hour in (6, 12, 18):
        _scheduler.add_job(
            _run_scheduled_prediction,
            trigger="cron",
            hour=hour,
            minute=0,
            id=f"predict_{hour:02d}",
        )
    _scheduler.start()
    print("[Monitor] 스케줄러 시작 — 06:00 / 12:00 / 18:00 자동 예측 등록")


def stop():
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        print("[Monitor] 스케줄러 종료")
