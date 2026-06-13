"""
멀티스텝 Q&A 에이전트 — Claude API (의도분류·답변생성) + 자체 LSTM·RAG·KG (예측 핵심)

도구: tool_lookup_actual(과거CSV) / tool_predict(LSTM) / tool_rag_search / tool_kg_query / tool_explain / tool_compare
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

# 폴백용 키워드 (Claude API 실패 시)
KEYWORDS = {
    "greet":   ["안녕", "반가워", "하이", "hello"],
    "help":    ["도움말", "뭐 할 수 있", "기능", "사용법"],
    "lookup":  ["실제", "실적", "기록", "있었어", "얼마였", "었나", "였어"],
    "predict": ["예측", "발전량", "얼마", "오늘", "내일", "이번", "출력", "생산"],
    "explain": ["왜", "이유", "원인", "설명", "어떻게", "영향", "기여"],
    "compare": ["비교", "차이", "vs", "대비", "더 높", "더 낮"],
    "rag":     ["과거", "유사", "비슷", "이전", "역대", "사례"],
    "kg":      ["태풍", "전선", "이상기후", "기상 이변", "규칙"],
    "risk":    ["위험", "경보", "풍속", "전운량", "급변", "기압", "폭풍"],
}

SYSTEM_PROMPT = """당신은 제주도 신재생에너지 발전량 예측 시스템의 친절한 AI 어시스턴트입니다.
- 수치는 반드시 포함하되 대화체로 자연스럽게 3~5문장으로 답변하세요
- 이모지를 적절히 활용하세요 (☀️ 💨 ⚡ ✅ ⚠️)
- 실적 데이터면 "예측"이 아닌 "실제 발전량"으로 표현하세요"""


class AgenticChatService:
    def __init__(self, agent, explainer):
        self.agent    = agent
        self.explainer = explainer
        self.history: List[Dict] = []   # 대화 히스토리 (최근 10턴)
        self.last_report    = None
        self.last_timestamp = None
        self.last_window    = None
        self._df = self._load_csv()

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

        # 1. 의도 + 날짜 추출 (Claude API)
        parsed    = self._claude_classify(message)
        intents   = parsed.get("intents", ["predict"])
        timestamp = parsed.get("timestamp") or self._parse_ts(message)
        trace += [{"tool": "intent_classifier", "result": intents},
                  {"tool": "time_parser",        "result": timestamp}]

        report, explanation, actual, compare_result = None, None, None, None
        is_past = self._is_past(timestamp)

        # 인사·도움말은 바로 답변
        if set(intents) <= {"greet", "help"}:
            answer = self._claude_answer(message, intents, timestamp, None, None, None, None)
            self._push_history(message, answer)
            return {"answer": answer, "tool_trace": trace, "report": None, "explanation": None}

        # Step A: 과거 → CSV / 미래 → LSTM
        if any(i in intents for i in ["predict","explain","compare","risk","lookup"]):
            if is_past:
                actual, t = self._tool_lookup(timestamp); trace.append(t)
                if actual:
                    report = {"predictions": {"solar_mw": actual["solar_mw"],
                                              "wind_mw":  actual["wind_mw"]},
                              "confidence": "High", "warnings": [], "similar_cases": [],
                              "kg_context": {"events":[], "rules":[]},
                              "trigger_active": False, "is_actual": True}
            else:
                window = self._build_window(timestamp)
                self.last_window = window
                report, t = self._tool_predict(window, timestamp); trace.append(t)
                if report: report["is_actual"] = False

        # Step B: RAG
        w = self.last_window if self.last_window is not None else self._build_window(timestamp)
        if "rag" in intents or (report and not report.get("similar_cases")):
            cases, t = self._tool_rag(w); trace.append(t)
            if report: report["similar_cases"] = cases

        # Step C: KG
        if "kg" in intents or "risk" in intents:
            tgt = report["similar_cases"][0]["timestamp"] if report and report.get("similar_cases") else timestamp
            kg, t = self._tool_kg(tgt); trace.append(t)
            if report: report["kg_context"] = kg

        # Step D: 설명
        if report and any(i in intents for i in ["predict","explain","risk"]):
            explanation, t = self._tool_explain(w, report); trace.append(t)

        # Step E: 비교
        if "compare" in intents:
            compare_result, t = self._tool_compare(message, report); trace.append(t)

        # 2. Claude API로 자연어 답변
        answer = self._claude_answer(message, intents, timestamp, report, explanation, compare_result, actual)
        self._push_history(message, answer)
        if report: self.last_report = report; self.last_timestamp = timestamp

        return {"answer": answer, "tool_trace": trace, "report": report, "explanation": explanation}

    # ── Claude API ────────────────────────────────────────────────────────────

    def _claude_classify(self, message: str) -> Dict:
        today = datetime.now().strftime("%Y-%m-%d")
        tmrw  = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        ctx   = f"직전 대화 시점: {self.last_timestamp}" if self.last_timestamp else ""
        prompt = (
            f"오늘: {today}. {ctx}\n"
            f"메시지: \"{message}\"\n\n"
            f"JSON만 반환 (설명 금지):\n"
            f'{{ "intents": [<greet|help|lookup|predict|explain|compare|rag|kg|risk>], '
            f'"timestamp": "<YYYY-MM-DD HH:MM:SS|null>" }}\n\n'
            f"날짜규칙: 오늘→{today} 14:00:00 / 내일→{tmrw} 14:00:00 / "
            f"시간없으면 14:00:00 / 날짜없으면 null"
        )
        try:
            res  = _claude.messages.create(model=MODEL, max_tokens=150,
                                           messages=[{"role":"user","content":prompt}])
            text = re.sub(r"```json|```", "", res.content[0].text).strip()
            return json.loads(text)
        except Exception as e:
            print(f"Claude 분류 실패: {e}")
            return {"intents": self._kw_classify(message), "timestamp": self._parse_ts(message)}

    def _claude_answer(self, message, intents, timestamp, report, explanation, compare_result, actual) -> str:
        summary = self._summarize(timestamp, report, explanation, compare_result, actual)
        msgs = [{"role": h["role"], "content": h["content"]} for h in self.history[-8:]]
        msgs.append({"role": "user", "content": f"질문: {message}\n\n데이터:\n{summary}"})
        try:
            res = _claude.messages.create(model=MODEL, max_tokens=400,
                                          system=SYSTEM_PROMPT, messages=msgs)
            return res.content[0].text.strip()
        except Exception as e:
            print(f"Claude 답변 실패: {e}")
            return self._fallback_answer(intents, timestamp, report, actual, compare_result, explanation)

    def _summarize(self, timestamp, report, explanation, compare_result, actual) -> str:
        p = []
        if actual:
            p += [f"[실제발전량({timestamp})]",
                  f"태양광:{actual['solar_mw']:.2f}MW / 풍력:{actual['wind_mw']:.2f}MW / "
                  f"총:{actual['solar_mw']+actual['wind_mw']:.2f}MW"]
        elif report:
            s, w = report["predictions"]["solar_mw"], report["predictions"]["wind_mw"]
            p += [f"[예측({timestamp})] 태양광:{s:.2f}MW 풍력:{w:.2f}MW 총:{s+w:.2f}MW",
                  f"신뢰도:{report.get('confidence','High')}"]
            if report.get("warnings"): p.append(f"경고:{','.join(report['warnings'])}")
            ref = report.get("refined_report")
            if ref and ref.get("refinement_applied"):
                lo, hi = ref.get("range_low",{}), ref.get("range_high",{})
                p.append(f"재예측범위: 태양광{lo.get('solar_mw',0):.1f}~{hi.get('solar_mw',0):.1f}MW "
                         f"풍력{lo.get('wind_mw',0):.1f}~{hi.get('wind_mw',0):.1f}MW")
        if explanation:
            top = explanation.get("variable_contributions",[])[:2]
            if top: p.append("영향변수:" + ",".join(f"{v['variable']}({v['impact_level']})" for v in top))
            for k in ("solar_explanation","wind_explanation","rag_comparison"):
                if explanation.get(k): p.append(explanation[k])
        if compare_result:
            ts2 = compare_result.get("timestamp2","")
            r2  = compare_result.get("report2",{}).get("predictions",{})
            sd, wd = compare_result.get("solar_diff",0), compare_result.get("wind_diff",0)
            p.append(f"[비교({ts2})] 태양광:{r2.get('solar_mw',0):.2f}MW({sd:+.1f}) "
                     f"풍력:{r2.get('wind_mw',0):.2f}MW({wd:+.1f})")
        return "\n".join(p) if p else "분석 데이터 없음"

    # ── 도구 ──────────────────────────────────────────────────────────────────

    def _tool_lookup(self, timestamp: str):
        """CSV 실적 조회"""
        try:
            if self._df is None: raise ValueError("CSV 미로드")
            dt   = datetime.strptime(timestamp[:10], "%Y-%m-%d")
            hour = int(timestamp[11:13]) if len(timestamp) > 10 else 14
            row  = self._df[(self._df["date"].dt.date == dt.date()) & (self._df["hour"] == hour)]
            if row.empty:
                day = self._df[self._df["date"].dt.date == dt.date()]
                if day.empty: return None, {"tool":"tool_lookup","status":"not_found"}
                result = {"solar_mw": float(day["solar_mw"].sum()),
                          "wind_mw":  float(day["wind_mw"].sum()), "type":"daily"}
            else:
                result = {"solar_mw": float(row.iloc[0]["solar_mw"]),
                          "wind_mw":  float(row.iloc[0]["wind_mw"]), "type":"hourly"}
            return result, {"tool":"tool_lookup","status":"ok","type":result["type"]}
        except Exception as e:
            return None, {"tool":"tool_lookup","status":"error","error":str(e)}

    def _tool_predict(self, window: np.ndarray, timestamp: str):
        try:
            fn = getattr(self.agent, "run_pipeline_with_refinement", self.agent.run_pipeline)
            report = fn(window, timestamp=timestamp)
            return report, {"tool":"tool_predict","status":"ok","timestamp":timestamp}
        except Exception as e:
            return ({"predictions":{"solar_mw":0.0,"wind_mw":0.0},"confidence":"Low",
                     "warnings":[str(e)],"similar_cases":[],"kg_context":{"events":[],"rules":[]},
                     "trigger_active":False},
                    {"tool":"tool_predict","status":"error","error":str(e)})

    def _tool_rag(self, window: np.ndarray):
        try:
            w = self.agent.feature_scaler.transform(window) if self.agent.feature_scaler else window.copy()
            cases = self.agent.rag_service.get_similar_cases(w.reshape(1,-1), top_k=5)
            return cases, {"tool":"tool_rag","status":"ok","found":len(cases)}
        except Exception as e:
            return [], {"tool":"tool_rag","status":"error","error":str(e)}

    def _tool_kg(self, timestamp: str):
        try:
            r = self.agent.kg_service.query_similar_events(timestamp)
            return r, {"tool":"tool_kg","status":"ok"}
        except Exception as e:
            return {"events":[],"rules":[]}, {"tool":"tool_kg","status":"error","error":str(e)}

    def _tool_explain(self, window: np.ndarray, report: dict):
        try:
            exp = self.explainer.explain(window, report, refined_report=report.get("refined_report"))
            return exp, {"tool":"tool_explain","status":"ok"}
        except Exception as e:
            return None, {"tool":"tool_explain","status":"error","error":str(e)}

    def _tool_compare(self, message: str, base: Optional[dict]):
        try:
            dates = re.findall(r"\d{4}-\d{2}-\d{2}", message)
            hours = re.findall(r"(\d{1,2})시", message)
            if len(dates) >= 2:
                ts2 = f"{dates[1]} {int(hours[1] if len(hours)>1 else 14):02d}:00:00"
            elif dates:
                ts2 = f"{dates[0]} {int(hours[0] if hours else 14):02d}:00:00"
            else:
                ts2 = (datetime.now()+timedelta(days=1)).strftime("%Y-%m-%d 14:00:00")
            r2 = self.agent.run_pipeline(self._build_window(ts2), timestamp=ts2)
            bs = base["predictions"]["solar_mw"] if base else 0
            bw = base["predictions"]["wind_mw"]  if base else 0
            return ({"timestamp2":ts2,"report2":r2,
                     "solar_diff":r2["predictions"]["solar_mw"]-bs,
                     "wind_diff": r2["predictions"]["wind_mw"]-bw},
                    {"tool":"tool_compare","status":"ok"})
        except Exception as e:
            return None, {"tool":"tool_compare","status":"error","error":str(e)}

    # ── 유틸 ──────────────────────────────────────────────────────────────────

    def _is_past(self, ts: str) -> bool:
        try: return datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S") < datetime.now()
        except: return False

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
        elif "지난달" in text or "지난 달" in text: now = (now.replace(day=1)-timedelta(days=1)).replace(day=1)
        elif "지난주" in text or "지난 주" in text: now -= timedelta(weeks=1)
        h = re.search(r"(\d{1,2})시", text)
        return now.strftime(f"%Y-%m-%d {int(h.group(1)) if h else 14:02d}:00:00")

    def _build_window(self, timestamp: str) -> np.ndarray:
        try: dt = datetime.strptime(timestamp[:19], "%Y-%m-%d %H:%M:%S")
        except: dt = datetime.now()
        rng = np.random.default_rng(int(hashlib.md5(timestamp.encode()).hexdigest(),16) % 10000)
        rows = []
        for i in range(24):
            th = (dt.hour - 23 + i) % 24
            if 6 <= th <= 18:
                sun = float(np.clip(np.sin(np.pi*(th-6)/12)*0.9 + rng.uniform(-0.05,0.05), 0, 1))
            else:
                sun = 0.0
            ws = float(max(0.5, rng.uniform(2,16) + np.sin(i/24*np.pi*2)*2.5 + rng.uniform(-0.5,0.5)))
            wa = rng.uniform(0, 2*np.pi)
            rows.append([ws, float(np.sin(wa)), float(np.cos(wa)),
                         float(20+np.sin((th-8)/24*np.pi*2)*5+rng.uniform(-0.5,0.5)),
                         float(60-np.sin((th-8)/24*np.pi*2)*15+rng.uniform(-2,2)),
                         float(1010+rng.uniform(-5,5)), float(rng.integers(0,10)),
                         sun, float(sun*2.8)])
        return np.array(rows)

    def _kw_classify(self, message: str) -> List[str]:
        found = [i for i, kws in KEYWORDS.items() if any(k in message for k in kws)]
        return found if found else ["greet"]

    def _push_history(self, message: str, answer: str):
        self.history += [{"role":"user","content":message},{"role":"assistant","content":answer}]
        if len(self.history) > 20: self.history = self.history[-20:]

    def _fallback_answer(self, intents, timestamp, report, actual, compare_result, explanation) -> str:
        if set(intents) <= {"greet"}: return "안녕하세요! 제주 신재생에너지 발전량 챗봇입니다 😊 발전량 예측, 과거 실적 조회, 비교 등을 물어보세요!"
        if set(intents) <= {"help"}:  return "• 오늘 발전량 예측해줘\n• 2025년 8월 15일 실적 얼마야?\n• 내일이랑 오늘 비교해줘\n• 왜 이렇게 예측했어?"
        p = []
        if actual:
            p.append(f"📊 실제발전량({timestamp})\n☀️ {actual['solar_mw']:.2f}MW | 💨 {actual['wind_mw']:.2f}MW | ⚡ {actual['solar_mw']+actual['wind_mw']:.2f}MW")
        elif report:
            s,w = report["predictions"]["solar_mw"], report["predictions"]["wind_mw"]
            e = {"High":"✅","Medium":"⚠️","Low":"🔴"}.get(report.get("confidence","High"),"")
            p.append(f"🔮 예측({timestamp})\n☀️ {s:.2f}MW | 💨 {w:.2f}MW | ⚡ {s+w:.2f}MW {e}")
        if explanation and "explain" in intents:
            p.append(explanation.get("full_narrative",""))
        if compare_result:
            ts2,sd,wd = compare_result["timestamp2"],compare_result["solar_diff"],compare_result["wind_diff"]
            r2 = compare_result["report2"]["predictions"]
            p.append(f"🔍 {ts2}: ☀️{r2['solar_mw']:.2f}MW({sd:+.1f}) 💨{r2['wind_mw']:.2f}MW({wd:+.1f})")
        return "\n".join(p) if p else "죄송해요, 다시 질문해주세요."