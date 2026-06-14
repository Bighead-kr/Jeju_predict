"""
멀티스텝 에이전트 — Claude function calling 기반 동적 도구 선택
도구: tool_lookup / tool_predict / tool_rag_search / tool_kg_query / tool_explain / tool_compare
"""
import re, os, json, hashlib
import numpy as np
import pandas as pd
import anthropic
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

_claude     = anthropic.Anthropic()
MODEL       = "claude-sonnet-4-6"
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH    = os.path.join(BASE_DIR, "docs", "jeju_solar_wind_generation.csv")

SYSTEM_PROMPT = """당신은 제주도 신재생에너지 발전량 예측 시스템의 AI 에이전트입니다.
- 제공된 도구를 활용해 질문에 필요한 데이터를 먼저 수집한 뒤 답변하세요
- 수치는 반드시 포함하되 대화체로 자연스럽게 3~5문장으로 답변하세요
- 이모지를 적절히 활용하세요 (☀️ 💨 ⚡ ✅ ⚠️)
- 실적 데이터면 "예측"이 아닌 "실제 발전량"으로 표현하세요
- 실측 데이터 범위: ~2025-12-31 (이 범위→tool_lookup, 이후→tool_predict)"""

# 폴백용 키워드 (Claude API 실패 시)
_KW = {
    "greet":   ["안녕", "반가워", "하이", "hello"],
    "help":    ["도움말", "뭐 할 수 있", "기능", "사용법"],
}

TOOLS = [
    {
        "name": "tool_lookup",
        "description": (
            "CSV에서 실제 발전량 실적 데이터를 조회합니다. "
            "2025-12-31 이전 날짜의 실제(과거) 발전량을 물어볼 때 사용하세요."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "timestamp": {
                    "type": "string",
                    "description": "조회할 날짜/시간 (YYYY-MM-DD HH:MM:SS 형식)"
                }
            },
            "required": ["timestamp"]
        }
    },
    {
        "name": "tool_predict",
        "description": (
            "LSTM 모델로 발전량을 예측합니다. "
            "미래 날짜나 실측 데이터가 없는 시점의 발전량 예측에 사용하세요."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "timestamp": {
                    "type": "string",
                    "description": "예측할 날짜/시간 (YYYY-MM-DD HH:MM:SS 형식)"
                }
            },
            "required": ["timestamp"]
        }
    },
    {
        "name": "tool_rag_search",
        "description": (
            "유사한 과거 기상 사례를 벡터 검색합니다. "
            "과거 유사 사례나 비슷한 기상 조건을 참고할 때 사용하세요."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "timestamp": {
                    "type": "string",
                    "description": "기준 날짜/시간 (YYYY-MM-DD HH:MM:SS 형식)"
                }
            },
            "required": ["timestamp"]
        }
    },
    {
        "name": "tool_kg_query",
        "description": (
            "지식 그래프에서 기상 이벤트와 규칙을 조회합니다. "
            "태풍, 이상기후, 기상 이변 등 특수 기상 상황 분석 시 사용하세요."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "timestamp": {
                    "type": "string",
                    "description": "조회할 날짜/시간 (YYYY-MM-DD HH:MM:SS 형식)"
                }
            },
            "required": ["timestamp"]
        }
    },
    {
        "name": "tool_explain",
        "description": (
            "발전량 예측 결과의 원인과 영향 변수를 설명합니다. "
            "왜 이런 예측이 나왔는지, 어떤 기상 요인이 영향을 줬는지 설명이 필요할 때 사용하세요. "
            "반드시 tool_predict 또는 tool_lookup을 먼저 호출한 뒤 사용하세요."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "timestamp": {
                    "type": "string",
                    "description": "설명할 예측의 날짜/시간 (YYYY-MM-DD HH:MM:SS 형식)"
                }
            },
            "required": ["timestamp"]
        }
    },
    {
        "name": "tool_compare",
        "description": (
            "두 시점의 발전량을 비교합니다. "
            "날짜 간 발전량 차이나 비교를 요청할 때 사용하세요."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "timestamp1": {
                    "type": "string",
                    "description": "첫 번째 비교 날짜/시간 (YYYY-MM-DD HH:MM:SS 형식)"
                },
                "timestamp2": {
                    "type": "string",
                    "description": "두 번째 비교 날짜/시간 (YYYY-MM-DD HH:MM:SS 형식)"
                }
            },
            "required": ["timestamp1", "timestamp2"]
        }
    },
]


class AgenticChatService:
    DEFAULT_ACTUAL_CUTOFF = datetime(2025, 12, 31, 23, 59, 59)

    def __init__(self, agent, explainer):
        self.agent     = agent
        self.explainer = explainer
        self.history: List[Dict] = []
        self.last_report    = None
        self.last_timestamp = None
        self.last_window    = None
        self._df = self._load_csv()
        if self._df is not None and not self._df.empty:
            self.actual_cutoff = datetime.combine(self._df["date"].max().date(),
                                                  datetime.max.time())
        else:
            self.actual_cutoff = self.DEFAULT_ACTUAL_CUTOFF

    def _load_csv(self) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
            df.columns = ["date", "hour", "solar_mw", "wind_mw"]
            df["date"] = pd.to_datetime(df["date"], format="%Y.%m.%d")
            df["hour"] = df["hour"].astype(int)
            print(f"CSV 로드: {len(df)}행 ({df['date'].min().date()} ~ {df['date'].max().date()})")
            return df
        except Exception as e:
            print(f"CSV 로드 실패: {e}")
            return None

    # ── 공개 인터페이스 ───────────────────────────────────────────────────────

    def chat(self, message: str) -> Dict[str, Any]:
        trace = []
        today = datetime.now().strftime("%Y-%m-%d")
        tmrw  = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        system = (
            f"{SYSTEM_PROMPT}\n"
            f"오늘: {today} / 내일: {tmrw}\n"
            + (f"직전 대화 시점: {self.last_timestamp}" if self.last_timestamp else "")
        )

        messages = [{"role": h["role"], "content": h["content"]} for h in self.history[-8:]]
        messages.append({"role": "user", "content": message})

        # 턴 내 공유 상태
        state: Dict[str, Any] = {
            "report":      self.last_report,
            "window":      self.last_window,
            "actual":      None,
            "explanation": None,
        }

        answer = ""
        for _ in range(6):  # 최대 6 스텝
            try:
                response = _claude.messages.create(
                    model=MODEL, max_tokens=800,
                    system=system, tools=TOOLS, messages=messages
                )
            except Exception as e:
                print(f"Claude API 오류: {e}")
                answer = self._fallback_answer(state)
                break

            if response.stop_reason == "end_turn":
                answer = next((b.text for b in response.content if hasattr(b, "text")), "")
                break

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    result_str = self._execute_tool(block.name, block.input, state, trace)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })
                    ts = block.input.get("timestamp") or block.input.get("timestamp1")
                    if ts:
                        self.last_timestamp = ts
                messages.append({"role": "user", "content": tool_results})
            else:
                answer = next((b.text for b in response.content if hasattr(b, "text")), "")
                break

        self._push_history(message, answer)
        if state.get("report"):
            self.last_report = state["report"]
        if state.get("window") is not None:
            self.last_window = state["window"]

        return {
            "answer":      answer,
            "tool_trace":  trace,
            "report":      state.get("report"),
            "explanation": state.get("explanation"),
        }

    # ── 도구 실행 디스패처 ────────────────────────────────────────────────────

    def _execute_tool(self, name: str, inputs: dict, state: dict, trace: list) -> str:
        """도구 실행 → JSON 문자열 반환 (tool_result content)"""
        timestamp = (
            inputs.get("timestamp")
            or self.last_timestamp
            or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        if name == "tool_lookup":
            actual, t = self._tool_lookup(timestamp)
            trace.append(t)
            if actual:
                state["actual"] = actual
                state["window"] = self._build_window(timestamp)
                state["report"] = {
                    "predictions": {"solar_mw": actual["solar_mw"], "wind_mw": actual["wind_mw"]},
                    "confidence": "High", "warnings": [], "similar_cases": [],
                    "kg_context": {"events": [], "rules": []},
                    "trigger_active": False, "is_actual": True,
                }
                return json.dumps({
                    "status": "ok", "timestamp": timestamp,
                    "solar_mw": actual["solar_mw"], "wind_mw": actual["wind_mw"],
                    "total_mw": actual["solar_mw"] + actual["wind_mw"],
                    "type": actual.get("type", "hourly"),
                }, ensure_ascii=False)
            return json.dumps({"status": "not_found",
                               "message": "해당 날짜의 실측 데이터가 없습니다."}, ensure_ascii=False)

        elif name == "tool_predict":
            window = self._build_window(timestamp)
            state["window"] = window
            report, t = self._tool_predict(window, timestamp)
            trace.append(t)
            state["report"] = report
            preds = report.get("predictions", {})
            solar, wind = preds.get("solar_mw", 0.0), preds.get("wind_mw", 0.0)
            return json.dumps({
                "status": "ok", "timestamp": timestamp,
                "solar_mw": solar, "wind_mw": wind, "total_mw": solar + wind,
                "confidence": report.get("confidence", "High"),
                "warnings": report.get("warnings", []),
            }, ensure_ascii=False)

        elif name == "tool_rag_search":
            window = state.get("window") or self._build_window(timestamp)
            cases, t = self._tool_rag(window)
            trace.append(t)
            if state.get("report"):
                state["report"]["similar_cases"] = cases
            top = [{"timestamp": c.get("timestamp", ""), "solar_mw": c.get("solar_mw", 0),
                    "wind_mw": c.get("wind_mw", 0), "similarity": c.get("similarity", 0)}
                   for c in cases[:3]]
            return json.dumps({"status": "ok", "cases_found": len(cases),
                               "top_cases": top}, ensure_ascii=False)

        elif name == "tool_kg_query":
            kg, t = self._tool_kg(timestamp)
            trace.append(t)
            if state.get("report"):
                state["report"]["kg_context"] = kg
            return json.dumps({
                "status": "ok",
                "events": kg.get("events", [])[:3],
                "rules":  kg.get("rules",  [])[:3],
            }, ensure_ascii=False)

        elif name == "tool_explain":
            window = state.get("window") or self._build_window(timestamp)
            report = state.get("report")
            if not report:
                return json.dumps({"status": "error",
                                   "message": "먼저 tool_predict 또는 tool_lookup을 호출하세요."}, ensure_ascii=False)
            explanation, t = self._tool_explain(window, report)
            trace.append(t)
            state["explanation"] = explanation
            if not explanation:
                return json.dumps({"status": "error", "message": "설명 생성 실패"}, ensure_ascii=False)
            top = explanation.get("variable_contributions", [])[:3]
            return json.dumps({
                "status": "ok",
                "solar_explanation": explanation.get("solar_explanation", ""),
                "wind_explanation":  explanation.get("wind_explanation", ""),
                "top_variables": [{"variable": v.get("variable"), "impact": v.get("impact_level")} for v in top],
            }, ensure_ascii=False)

        elif name == "tool_compare":
            ts1 = inputs.get("timestamp1", timestamp)
            ts2 = inputs.get("timestamp2", "")
            if not ts2:
                return json.dumps({"status": "error",
                                   "message": "timestamp2가 필요합니다."}, ensure_ascii=False)
            base = state.get("report")
            if not base:
                w1 = self._build_window(ts1)
                base, t = self._tool_predict(w1, ts1)
                trace.append(t)
            w2 = self._build_window(ts2)
            r2, t = self._tool_predict(w2, ts2)
            trace.append(t)
            bs = base["predictions"]["solar_mw"]
            bw = base["predictions"]["wind_mw"]
            s2 = r2["predictions"]["solar_mw"]
            w2v = r2["predictions"]["wind_mw"]
            return json.dumps({
                "status": "ok",
                "timestamp1": ts1, "solar1": bs, "wind1": bw, "total1": bs + bw,
                "timestamp2": ts2, "solar2": s2, "wind2": w2v, "total2": s2 + w2v,
                "solar_diff": s2 - bs, "wind_diff": w2v - bw,
                "total_diff": (s2 + w2v) - (bs + bw),
            }, ensure_ascii=False)

        return json.dumps({"status": "error", "message": f"알 수 없는 도구: {name}"}, ensure_ascii=False)

    # ── 도구 구현 ─────────────────────────────────────────────────────────────

    def _tool_lookup(self, timestamp: str):
        try:
            if self._df is None: raise ValueError("CSV 미로드")
            dt   = datetime.strptime(timestamp[:10], "%Y-%m-%d")
            hour = int(timestamp[11:13]) if len(timestamp) > 10 else 14
            row  = self._df[(self._df["date"].dt.date == dt.date()) & (self._df["hour"] == hour)]
            if row.empty:
                day = self._df[self._df["date"].dt.date == dt.date()]
                if day.empty: return None, {"tool": "tool_lookup", "status": "not_found"}
                result = {"solar_mw": float(day["solar_mw"].sum()),
                          "wind_mw":  float(day["wind_mw"].sum()), "type": "daily"}
            else:
                result = {"solar_mw": float(row.iloc[0]["solar_mw"]),
                          "wind_mw":  float(row.iloc[0]["wind_mw"]), "type": "hourly"}
            return result, {"tool": "tool_lookup", "status": "ok", "type": result["type"]}
        except Exception as e:
            return None, {"tool": "tool_lookup", "status": "error", "error": str(e)}

    def _tool_predict(self, window: np.ndarray, timestamp: str):
        try:
            fn = getattr(self.agent, "run_pipeline_with_refinement", self.agent.run_pipeline)
            report = fn(window, timestamp=timestamp)
            return report, {"tool": "tool_predict", "status": "ok", "timestamp": timestamp}
        except Exception as e:
            return ({"predictions": {"solar_mw": 0.0, "wind_mw": 0.0}, "confidence": "Low",
                     "warnings": [str(e)], "similar_cases": [],
                     "kg_context": {"events": [], "rules": []}, "trigger_active": False},
                    {"tool": "tool_predict", "status": "error", "error": str(e)})

    def _tool_rag(self, window: np.ndarray):
        try:
            w = self.agent.feature_scaler.transform(window) if self.agent.feature_scaler else window.copy()
            cases = self.agent.rag_service.get_similar_cases(w.reshape(1, -1), top_k=5)
            return cases, {"tool": "tool_rag", "status": "ok", "found": len(cases)}
        except Exception as e:
            return [], {"tool": "tool_rag", "status": "error", "error": str(e)}

    def _tool_kg(self, timestamp: str):
        try:
            r = self.agent.kg_service.query_similar_events(timestamp)
            return r, {"tool": "tool_kg", "status": "ok"}
        except Exception as e:
            return {"events": [], "rules": []}, {"tool": "tool_kg", "status": "error", "error": str(e)}

    def _tool_explain(self, window: np.ndarray, report: dict):
        try:
            exp = self.explainer.explain(window, report, refined_report=report.get("refined_report"))
            return exp, {"tool": "tool_explain", "status": "ok"}
        except Exception as e:
            return None, {"tool": "tool_explain", "status": "error", "error": str(e)}

    # ── 유틸 ──────────────────────────────────────────────────────────────────

    def _is_within_actual(self, ts: str) -> bool:
        try:
            return datetime.strptime(ts[:10], "%Y-%m-%d") <= self.actual_cutoff
        except:
            return False

    def _parse_ts(self, text: str) -> str:
        now = datetime.now()
        for pat, fmt in [(r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})", lambda m: f"{m.group(1)} {m.group(2)}:00"),
                         (r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일",
                          lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d} 14:00:00"),
                         (r"(\d{1,2})월\s*(\d{1,2})일",
                          lambda m: f"{now.year}-{int(m.group(1)):02d}-{int(m.group(2)):02d} 14:00:00")]:
            m = re.search(pat, text)
            if m: return fmt(m)
        if "내일" in text: now += timedelta(days=1)
        elif "어제" in text: now -= timedelta(days=1)
        elif "이번달" in text or "이번 달" in text: now = now.replace(day=1)
        elif "지난달" in text or "지난 달" in text: now = (now.replace(day=1) - timedelta(days=1)).replace(day=1)
        elif "지난주" in text or "지난 주" in text: now -= timedelta(weeks=1)
        h = re.search(r"(\d{1,2})시", text)
        return now.strftime(f"%Y-%m-%d {int(h.group(1)) if h else 14:02d}:00:00")

    def _build_window(self, timestamp: str) -> np.ndarray:
        try: dt = datetime.strptime(timestamp[:19], "%Y-%m-%d %H:%M:%S")
        except: dt = datetime.now()
        rng = np.random.default_rng(int(hashlib.md5(timestamp.encode()).hexdigest(), 16) % 10000)
        rows = []
        for i in range(24):
            th = (dt.hour - 23 + i) % 24
            if 6 <= th <= 18:
                sun = float(np.clip(np.sin(np.pi * (th - 6) / 12) * 0.9 + rng.uniform(-0.05, 0.05), 0, 1))
            else:
                sun = 0.0
            ws = float(max(0.5, rng.uniform(2, 16) + np.sin(i / 24 * np.pi * 2) * 2.5 + rng.uniform(-0.5, 0.5)))
            wa = rng.uniform(0, 2 * np.pi)
            rows.append([ws, float(np.sin(wa)), float(np.cos(wa)),
                         float(20 + np.sin((th - 8) / 24 * np.pi * 2) * 5 + rng.uniform(-0.5, 0.5)),
                         float(60 - np.sin((th - 8) / 24 * np.pi * 2) * 15 + rng.uniform(-2, 2)),
                         float(1010 + rng.uniform(-5, 5)), float(rng.integers(0, 10)),
                         sun, float(sun * 2.8)])
        return np.array(rows)

    def _push_history(self, message: str, answer: str):
        self.history += [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
        if len(self.history) > 20:
            self.history = self.history[-20:]

    def _fallback_answer(self, state: dict) -> str:
        report = state.get("report")
        actual = state.get("actual")
        if actual:
            return (f"📊 실제발전량\n"
                    f"☀️ {actual['solar_mw']:.2f}MW | 💨 {actual['wind_mw']:.2f}MW | "
                    f"⚡ {actual['solar_mw']+actual['wind_mw']:.2f}MW")
        if report:
            s, w = report["predictions"]["solar_mw"], report["predictions"]["wind_mw"]
            badge = {"High": "✅", "Medium": "⚠️", "Low": "🔴"}.get(report.get("confidence", "High"), "")
            return f"🔮 예측\n☀️ {s:.2f}MW | 💨 {w:.2f}MW | ⚡ {s+w:.2f}MW {badge}"
        return "죄송해요, 일시적 오류가 발생했습니다. 다시 질문해주세요."
