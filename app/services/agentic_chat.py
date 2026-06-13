"""
#3 멀티스텝 Q&A 에이전트 (Agentic RAG with Tool Use)

사용자 질문을 분석하여 필요한 도구를 단계적으로 선택·실행한 뒤 답변을 조합합니다.

도구 목록:
  - tool_predict    : LSTM 모델로 발전량 예측
  - tool_rag_search : FAISS로 유사 과거 사례 검색
  - tool_kg_query   : 지식 그래프에서 이벤트/규칙 조회
  - tool_explain    : 예측 결과에 대한 자연어 설명 생성
  - tool_compare    : 두 시점 예측 비교
"""

import re
import numpy as np
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any


# ── 의도(Intent) 분류 키워드 ─────────────────────────────────────────────────
INTENT_KEYWORDS = {
    "predict":  ["예측", "발전량", "얼마", "오늘", "내일", "다음", "이번", "출력", "생산"],
    "explain":  ["왜", "이유", "원인", "설명", "어떻게", "영향", "기여", "근거"],
    "compare":  ["비교", "차이", "vs", "대비", "더 높", "더 낮", "어느"],
    "rag":      ["과거", "유사", "비슷", "이전", "역대", "같은 날", "사례"],
    "kg":       ["태풍", "전선", "이상기후", "기상 이변", "규칙", "제약", "물리"],
    "risk":     ["위험", "경보", "주의", "풍속", "전운량", "급변", "기압", "폭풍"],
}


class AgenticChatService:
    """
    멀티스텝 Q&A 에이전트.
    agent(JejuPredictAgent)와 explainer(ExplainerService)를 외부에서 주입받습니다.
    """

    def __init__(self, agent, explainer):
        self.agent = agent
        self.explainer = explainer

    # ── 공개 인터페이스 ───────────────────────────────────────────────────────

    def chat(self, message: str) -> Dict[str, Any]:
        """
        Returns:
            {
              "answer": str,            # 최종 자연어 답변
              "tool_trace": list[dict], # 실행된 도구 목록 (디버깅/UI용)
              "report": dict | None,    # 마지막으로 생성된 예측 리포트
              "explanation": dict | None,
            }
        """
        tool_trace: List[dict] = []
        context: Dict[str, Any] = {}  # 도구 간 공유 컨텍스트

        # 1. 의도 분류
        intents = self._classify_intents(message)
        tool_trace.append({"tool": "intent_classifier", "result": intents})

        # 2. 타임스탬프 파싱
        timestamp, window = self._parse_time_and_build_window(message)
        tool_trace.append({"tool": "time_parser", "result": timestamp})

        # ── 도구 실행 체인 ─────────────────────────────────────────────────
        report = None
        explanation = None
        answer_parts: List[str] = []

        # Step A: 예측이 필요한 의도면 항상 먼저 예측 실행
        if any(i in intents for i in ["predict", "explain", "compare", "risk"]):
            report, trace = self._tool_predict(window, timestamp)
            tool_trace.append(trace)
            context["report"] = report
            context["window"] = window
            context["timestamp"] = timestamp

        # Step B: RAG 명시적 요청 또는 유사사례가 없을 때 보완
        if "rag" in intents or (report and not report.get("similar_cases")):
            rag_result, trace = self._tool_rag_search(window)
            tool_trace.append(trace)
            if report:
                report["similar_cases"] = rag_result

        # Step C: KG 조회 (기상 이변/위험 의도)
        if "kg" in intents or "risk" in intents:
            if report and report.get("similar_cases"):
                target_ts = report["similar_cases"][0]["timestamp"]
            else:
                target_ts = timestamp
            kg_result, trace = self._tool_kg_query(target_ts)
            tool_trace.append(trace)
            if report:
                report["kg_context"] = kg_result

        # Step D: 설명 에이전트 실행 (예측/설명/위험 의도)
        if report and any(i in intents for i in ["predict", "explain", "risk"]):
            explanation, trace = self._tool_explain(window, report)
            tool_trace.append(trace)
            context["explanation"] = explanation

        # Step E: 비교 의도 — 두 번째 시점 추출해 비교
        compare_result = None
        if "compare" in intents:
            compare_result, trace = self._tool_compare(message, report)
            tool_trace.append(trace)

        # 3. 최종 답변 조합
        answer = self._build_answer(
            message, intents, timestamp, report, explanation, compare_result
        )

        return {
            "answer": answer,
            "tool_trace": tool_trace,
            "report": report,
            "explanation": explanation,
        }

    # ── 도구 구현 ─────────────────────────────────────────────────────────────

    def _tool_predict(self, window: np.ndarray, timestamp: str):
        """도구: LSTM 파이프라인 예측"""
        try:
            report = self.agent.run_pipeline(window, timestamp=timestamp)
            # refined_report도 함께 실행 (자율 재예측 #1)
            if hasattr(self.agent, "run_pipeline_with_refinement"):
                report = self.agent.run_pipeline_with_refinement(window, timestamp=timestamp)
            return report, {"tool": "tool_predict", "status": "ok", "timestamp": timestamp}
        except Exception as e:
            fallback = {
                "predictions": {"solar_mw": 0.0, "wind_mw": 0.0},
                "confidence": "Low",
                "warnings": [str(e)],
                "similar_cases": [],
                "kg_context": {"events": [], "rules": []},
                "trigger_active": False,
            }
            return fallback, {"tool": "tool_predict", "status": "error", "error": str(e)}

    def _tool_rag_search(self, window: np.ndarray):
        """도구: FAISS 유사 사례 검색"""
        try:
            # 스케일링
            w_scaled = window.copy()
            if self.agent.feature_scaler is not None:
                w_scaled = self.agent.feature_scaler.transform(window)
            flat = w_scaled.reshape(1, -1)
            cases = self.agent.rag_service.get_similar_cases(flat, top_k=5)
            return cases, {"tool": "tool_rag_search", "status": "ok", "found": len(cases)}
        except Exception as e:
            return [], {"tool": "tool_rag_search", "status": "error", "error": str(e)}

    def _tool_kg_query(self, timestamp: str):
        """도구: KG 이벤트/규칙 조회"""
        try:
            result = self.agent.kg_service.query_similar_events(timestamp)
            return result, {"tool": "tool_kg_query", "status": "ok", "timestamp": timestamp}
        except Exception as e:
            return {"events": [], "rules": []}, {"tool": "tool_kg_query", "status": "error", "error": str(e)}

    def _tool_explain(self, window: np.ndarray, report: dict):
        """도구: 예측 설명 에이전트"""
        try:
            refined = report.get("refined_report")
            explanation = self.explainer.explain(window, report, refined_report=refined)
            return explanation, {"tool": "tool_explain", "status": "ok"}
        except Exception as e:
            return None, {"tool": "tool_explain", "status": "error", "error": str(e)}

    def _tool_compare(self, message: str, base_report: dict):
        """도구: 두 시점 비교 (질문에서 두 번째 시점 추출)"""
        try:
            dates = re.findall(r"\d{4}-\d{2}-\d{2}", message)
            hours = re.findall(r"\d{1,2}시|(\d{2}:\d{2})", message)

            if len(dates) >= 2:
                ts2 = f"{dates[1]} {hours[1] if len(hours) > 1 else '14'}:00:00"
            elif len(dates) == 1:
                ts2 = f"{dates[0]} {hours[-1] if hours else '09'}:00:00"
            else:
                # 내일 비교
                now = datetime.now()
                ts2 = (now + timedelta(days=1)).strftime("%Y-%m-%d 14:00:00")

            _, w2 = self._parse_time_and_build_window(ts2)
            report2 = self.agent.run_pipeline(w2, timestamp=ts2)

            base_solar = base_report["predictions"]["solar_mw"]
            base_wind = base_report["predictions"]["wind_mw"]
            cmp_solar = report2["predictions"]["solar_mw"]
            cmp_wind = report2["predictions"]["wind_mw"]

            compare_result = {
                "timestamp2": ts2,
                "report2": report2,
                "solar_diff": cmp_solar - base_solar,
                "wind_diff": cmp_wind - base_wind,
            }
            return compare_result, {"tool": "tool_compare", "status": "ok", "ts2": ts2}
        except Exception as e:
            return None, {"tool": "tool_compare", "status": "error", "error": str(e)}

    # ── 의도 분류 & 타임 파싱 ─────────────────────────────────────────────────

    def _classify_intents(self, message: str) -> List[str]:
        found = []
        for intent, keywords in INTENT_KEYWORDS.items():
            if any(kw in message for kw in keywords):
                found.append(intent)
        # 기본 폴백
        if not found:
            found = ["predict"]
        return found

    def _parse_time_and_build_window(self, text: str):
        """
        텍스트에서 타임스탬프를 추출하고 해당 시점의 24h 더미 기상 윈도우를 생성.
        """
        date_pattern = r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})"
        match = re.search(date_pattern, text)

        if match:
            timestamp = f"{match.group(1)} {match.group(2)}:00"
        else:
            # 오늘/내일 등 상대 표현 처리
            now = datetime.now()
            if "내일" in text:
                now += timedelta(days=1)
            elif "어제" in text:
                now -= timedelta(days=1)
            # 시간 추출 시도
            hour_match = re.search(r"(\d{1,2})시", text)
            hour = int(hour_match.group(1)) if hour_match else 14
            timestamp = now.strftime(f"%Y-%m-%d {hour:02d}:00:00")

        window = self._build_window(timestamp)
        return timestamp, window

    def _build_window(self, timestamp: str) -> np.ndarray:
        """타임스탬프 기반으로 결정론적 더미 24h 기상 윈도우 생성"""
        try:
            dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            dt = datetime.now()

        hour = dt.hour
        seed_val = int(hashlib.md5(timestamp.encode()).hexdigest(), 16) % 10000
        rng = np.random.default_rng(seed_val)  # 전역 시드 오염 없는 독립 RNG

        rows = []
        for i in range(24):
            t_hour = (hour - 23 + i) % 24

            # 일조/일사
            if 6 <= t_hour <= 18:
                angle = np.pi * (t_hour - 6) / 12
                sunshine = float(np.sin(angle) * 0.9 + rng.uniform(-0.05, 0.05))
                sunshine = max(0.0, min(1.0, sunshine))
                solar_rad = float(sunshine * 2.8)
            else:
                sunshine, solar_rad = 0.0, 0.0

            # 풍속/풍향
            base_wind = rng.uniform(2.0, 16.0)
            wind_speed = float(max(0.5, base_wind + np.sin(i / 24 * np.pi * 2) * 2.5 + rng.uniform(-0.5, 0.5)))
            wind_angle = rng.uniform(0, 2 * np.pi)

            temp = float(20.0 + np.sin((t_hour - 8) / 24 * np.pi * 2) * 5.0 + rng.uniform(-0.5, 0.5))
            humidity = float(60.0 - np.sin((t_hour - 8) / 24 * np.pi * 2) * 15.0 + rng.uniform(-2.0, 2.0))
            pressure = float(1010.0 + rng.uniform(-5.0, 5.0))
            cloud = float(rng.integers(0, 10))

            rows.append([
                wind_speed,
                float(np.sin(wind_angle)),
                float(np.cos(wind_angle)),
                temp, humidity, pressure, cloud, sunshine, solar_rad
            ])

        return np.array(rows)

    # ── 답변 조합 ─────────────────────────────────────────────────────────────

    def _build_answer(
        self,
        message: str,
        intents: List[str],
        timestamp: str,
        report: Optional[dict],
        explanation: Optional[dict],
        compare_result: Optional[dict],
    ) -> str:
        parts: List[str] = []

        parts.append(f"{timestamp} 기준 분석 결과입니다.\n")

        if report:
            solar = report["predictions"]["solar_mw"]
            wind = report["predictions"]["wind_mw"]
            conf = report.get("confidence", "High")
            conf_emoji = {"High": "✅", "Medium": "⚠️", "Low": "🔴"}.get(conf, "")
            parts.append(f"☀️ 태양광: {solar:.2f} MW | 💨 풍력: {wind:.2f} MW")
            parts.append(f"{conf_emoji} 신뢰도: {conf}\n")

            # 경고
            if report.get("warnings"):
                parts.append("⚠️ 기상 이상 감지")
                for w in report["warnings"]:
                    parts.append(f"  - {w}")
                parts.append("")

            # 불확실성 범위 (자율 재예측 결과)
            refined = report.get("refined_report")
            if refined and refined.get("refinement_applied"):
                low = refined.get("range_low", {})
                high = refined.get("range_high", {})
                if low and high:
                    parts.append(
                        f"🔁 자율 재예측 범위: "
                        f"태양광 {low['solar_mw']:.1f}~{high['solar_mw']:.1f} MW / "
                        f"풍력 {low['wind_mw']:.1f}~{high['wind_mw']:.1f} MW\n"
                    )

        # 설명 의도
        if explanation and "explain" in intents:
            parts.append("---")
            parts.append(explanation.get("full_narrative", ""))
        elif explanation:
            # 설명 요약만
            top_vars = explanation.get("variable_contributions", [])[:3]
            if top_vars:
                var_summary = ", ".join(
                    f"{v['variable']}({v['impact_level']})" for v in top_vars
                )
                parts.append(f"📊 주요 영향 변수: {var_summary}")
            solar_exp = explanation.get("solar_explanation", "")
            if solar_exp:
                parts.append(f"☀️ {solar_exp}")
            wind_exp = explanation.get("wind_explanation", "")
            if wind_exp:
                parts.append(f"💨 {wind_exp}")
            rag_cmp = explanation.get("rag_comparison", "")
            if rag_cmp:
                parts.append(f"📚 {rag_cmp}")
            parts.append("")

        # 비교 결과
        if compare_result:
            ts2 = compare_result.get("timestamp2", "")
            r2 = compare_result.get("report2", {})
            s2 = r2.get("predictions", {}).get("solar_mw", 0)
            w2 = r2.get("predictions", {}).get("wind_mw", 0)
            sd = compare_result.get("solar_diff", 0)
            wd = compare_result.get("wind_diff", 0)
            parts.append(f"---\n🔍 {ts2} 비교")
            parts.append(f"태양광: {s2:.2f} MW ({'▲' if sd >= 0 else '▼'}{abs(sd):.1f} MW 차이)")
            parts.append(f"풍력: {w2:.2f} MW ({'▲' if wd >= 0 else '▼'}{abs(wd):.1f} MW 차이)")

        return "\n".join(parts)
