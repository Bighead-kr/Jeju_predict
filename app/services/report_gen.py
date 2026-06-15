"""
A-7: 일간 운영 리포트 HTML/JSON 생성
- 24H 예측 요약 + 경보 이력 → reports/{date}.json / .html
"""
import os
import json
from datetime import datetime, timedelta
from typing import Any, Optional

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_DIR  = os.path.join(BASE_DIR, "data", "reports")


def _ensure_dir():
    os.makedirs(REPORTS_DIR, exist_ok=True)


def generate_report(
    date: str,
    hourly: list[dict],
    events: list[dict],
    is_actual: bool = False,
) -> dict[str, Any]:
    """
    date      : "YYYY-MM-DD"
    hourly    : [{"hour":"00","solar_mw":...,"wind_mw":...}, ...]
    events    : event_store.get_events() 필터링 결과
    is_actual : 실측 데이터 여부
    """
    _ensure_dir()

    solar_total = sum(h.get("solar_mw", 0) for h in hourly)
    wind_total  = sum(h.get("wind_mw",  0) for h in hourly)

    # 경보 이벤트만 필터
    alerts = [e for e in events if e.get("type") in ("trigger", "alert")]

    report: dict[str, Any] = {
        "date":         date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "is_actual":    is_actual,
        "summary": {
            "solar_total_mw": round(solar_total, 2),
            "wind_total_mw":  round(wind_total,  2),
            "total_mw":       round(solar_total + wind_total, 2),
            "alert_count":    len(alerts),
        },
        "hourly":  hourly,
        "alerts":  alerts,
        "events":  events,
    }

    # JSON 저장
    json_path = os.path.join(REPORTS_DIR, f"{date}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # HTML 저장
    html_path = os.path.join(REPORTS_DIR, f"{date}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(_render_html(report))

    print(f"[ReportGen] {date} 리포트 생성 완료 ({json_path})")
    return report


def load_report(date: str) -> Optional[dict]:
    """저장된 리포트 JSON 로드. 없으면 None."""
    _ensure_dir()
    path = os.path.join(REPORTS_DIR, f"{date}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _render_html(report: dict) -> str:
    s   = report["summary"]
    date = report["date"]
    rows = "".join(
        f"<tr><td>{h['hour']}시</td><td>{h.get('solar_mw',0):.2f}</td>"
        f"<td>{h.get('wind_mw',0):.2f}</td></tr>"
        for h in report.get("hourly", [])
    )
    alerts_html = "없음" if not report["alerts"] else "<ul>" + "".join(
        f"<li>[{a.get('timestamp','')}] {a.get('reason','')}</li>"
        for a in report["alerts"]
    ) + "</ul>"

    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<title>일간 리포트 — {date}</title>
<style>
  body{{font-family:sans-serif;margin:2rem;color:#222}}
  h1{{color:#1a56db}}table{{border-collapse:collapse;width:100%}}
  th,td{{border:1px solid #ddd;padding:6px 12px;text-align:right}}
  th{{background:#f0f4ff;text-align:center}}
</style></head><body>
<h1>제주 신재생에너지 일간 리포트</h1>
<p>날짜: <strong>{date}</strong> &nbsp;|&nbsp; 생성: {report['generated_at']} &nbsp;|&nbsp;
   데이터 유형: {'실측' if report['is_actual'] else '예측'}</p>
<h2>요약</h2>
<ul>
  <li>태양광 합계: <strong>{s['solar_total_mw']} MW</strong></li>
  <li>풍력 합계: <strong>{s['wind_total_mw']} MW</strong></li>
  <li>총 발전량: <strong>{s['total_mw']} MW</strong></li>
  <li>경보 건수: <strong>{s['alert_count']}건</strong></li>
</ul>
<h2>시간대별 발전량</h2>
<table><tr><th>시각</th><th>태양광 (MW)</th><th>풍력 (MW)</th></tr>
{rows}
</table>
<h2>경보 이력</h2>
{alerts_html}
</body></html>"""
