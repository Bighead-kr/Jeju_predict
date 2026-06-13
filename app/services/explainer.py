"""
#4 예측 설명 에이전트 (Explainability Agent)

예측 결과(solar_mw, wind_mw)에 대해 다음 3가지를 조합하여 자연어 설명 생성:
  A. 입력 변수 기여도 분석 (permutation importance 근사 + 직관 규칙)
  B. RAG 유사사례와의 비교
  C. KG 물리 제약 적용 여부

외부 의존성 없이 순수 numpy만 사용 (LLM 불필요).
"""

import numpy as np
from typing import Optional


# ── 변수 인덱스 & 이름 매핑 ──────────────────────────────────────────────────
FEATURE_NAMES = {
    0: ("풍속", "m/s"),
    1: ("풍향(sin)", ""),
    2: ("풍향(cos)", ""),
    3: ("기온", "°C"),
    4: ("습도", "%"),
    5: ("기압", "hPa"),
    6: ("전운량", ""),
    7: ("일조량", ""),
    8: ("일사량", "MJ/m²"),
}

# 태양광에 영향을 주는 변수 인덱스
SOLAR_RELEVANT = [7, 8, 6, 3]
# 풍력에 영향을 주는 변수 인덱스
WIND_RELEVANT = [0, 1, 2, 5]


class ExplainerService:
    """
    예측값에 대한 자연어 설명을 생성하는 서비스.
    모델 없이 입력 윈도우 통계 + RAG/KG 결과를 조합.
    """

    def explain(
        self,
        current_window: np.ndarray,
        report: dict,
        refined_report: Optional[dict] = None,
    ) -> dict:
        """
        Args:
            current_window: (24, 9) 기상 데이터
            report: run_pipeline()이 반환한 기본 리포트
            refined_report: 자율 재예측 결과 (있을 경우 불확실성 포함)

        Returns:
            {
              "solar_explanation": str,
              "wind_explanation": str,
              "variable_contributions": list[dict],
              "rag_comparison": str,
              "kg_rules_applied": list[str],
              "confidence_narrative": str,
              "full_narrative": str,
            }
        """
        solar_mw = report["predictions"]["solar_mw"]
        wind_mw = report["predictions"]["wind_mw"]
        warnings = report.get("warnings", [])
        similar_cases = report.get("similar_cases", [])
        kg_context = report.get("kg_context", {"events": [], "rules": []})
        confidence = report.get("confidence", "High")

        # ── A. 입력 변수 기여도 분석 ─────────────────────────────────────────
        contributions = self._compute_contributions(current_window)

        # ── B. RAG 유사사례 비교 ─────────────────────────────────────────────
        rag_comparison = self._build_rag_comparison(solar_mw, wind_mw, similar_cases)

        # ── C. KG 물리 규칙 적용 내역 ────────────────────────────────────────
        kg_rules_applied = self._extract_kg_rules(kg_context, current_window)

        # ── D. 개별 설명문 생성 ──────────────────────────────────────────────
        solar_explanation = self._explain_solar(solar_mw, current_window, contributions)
        wind_explanation = self._explain_wind(wind_mw, current_window, contributions)

        # ── E. 신뢰도 서술 ───────────────────────────────────────────────────
        confidence_narrative = self._explain_confidence(
            confidence, warnings, similar_cases, refined_report
        )

        # ── F. 종합 자연어 서술 ──────────────────────────────────────────────
        full_narrative = self._build_full_narrative(
            solar_mw, wind_mw,
            solar_explanation, wind_explanation,
            rag_comparison, kg_rules_applied,
            confidence_narrative, refined_report
        )

        return {
            "solar_explanation": solar_explanation,
            "wind_explanation": wind_explanation,
            "variable_contributions": contributions,
            "rag_comparison": rag_comparison,
            "kg_rules_applied": kg_rules_applied,
            "confidence_narrative": confidence_narrative,
            "full_narrative": full_narrative,
        }

    # ── 내부 메서드 ──────────────────────────────────────────────────────────

    def _compute_contributions(self, window: np.ndarray) -> list:
        """
        각 변수의 최근값(마지막 타임스텝)과 24시간 평균의 편차를 통해
        기여도를 근사적으로 계산합니다.
        Permutation importance를 모델 없이 대리하는 통계적 접근.
        """
        last = window[-1]       # 가장 최근 관측값
        mean = window.mean(axis=0)
        std = window.std(axis=0) + 1e-8  # 0 나누기 방지

        contributions = []
        for idx, (name, unit) in FEATURE_NAMES.items():
            deviation = (last[idx] - mean[idx]) / std[idx]  # z-score 편차
            direction = "상승" if deviation > 0 else "하강"
            magnitude = abs(deviation)

            # 기여 강도 등급
            if magnitude >= 1.5:
                level = "강한 영향"
            elif magnitude >= 0.7:
                level = "중간 영향"
            else:
                level = "미미한 영향"

            contributions.append({
                "variable": name,
                "unit": unit,
                "current_value": round(float(last[idx]), 3),
                "24h_mean": round(float(mean[idx]), 3),
                "z_score": round(float(deviation), 3),
                "direction": direction,
                "impact_level": level,
            })

        # z-score 절댓값 기준 내림차순
        contributions.sort(key=lambda x: abs(x["z_score"]), reverse=True)
        return contributions

    def _explain_solar(self, solar_mw: float, window: np.ndarray, contributions: list) -> str:
        last = window[-1]
        sunshine = last[7]
        cloud = last[6]
        solar_rad = last[8]

        if solar_mw <= 0.0:
            if sunshine <= 0.01:
                return "야간 시간대로 일조량이 0에 가까워 태양광 발전량이 0 MW로 억제되었습니다. (물리 제약 적용)"
            return f"전운량({cloud:.0f}/10)이 높아 태양광 발전이 불가 수준으로 제한되었습니다."

        # 상위 기여 변수 중 태양광 관련 추출
        top_solar_vars = [c for c in contributions if FEATURE_NAMES.get(
            next((k for k, v in FEATURE_NAMES.items() if v[0] == c["variable"]), -1),
            ("", ""))[0] in [FEATURE_NAMES[i][0] for i in SOLAR_RELEVANT]][:2]

        parts = [f"태양광 발전량은 {solar_mw:.2f} MW로 예측되었습니다."]

        if sunshine > 0.7:
            parts.append(f"일조량({sunshine:.2f})이 높아 발전 조건이 우수합니다.")
        elif sunshine > 0.3:
            parts.append(f"일조량({sunshine:.2f})이 보통 수준으로 평균적인 발전이 예상됩니다.")
        else:
            parts.append(f"일조량({sunshine:.2f})이 낮아 발전량이 제한됩니다.")

        if cloud >= 7:
            parts.append(f"전운량({cloud:.0f}/10)이 높아 일사를 차단 중입니다.")
        elif cloud <= 3:
            parts.append(f"맑은 하늘(전운량 {cloud:.0f}/10)로 태양광 효율이 극대화됩니다.")

        return " ".join(parts)

    def _explain_wind(self, wind_mw: float, window: np.ndarray, contributions: list) -> str:
        last = window[-1]
        wind_speed = last[0]
        pressure = last[5]
        pressure_24h_mean = window[:, 5].mean()
        pressure_change = pressure - pressure_24h_mean

        if wind_mw <= 0.0:
            if wind_speed < 3.0:
                return f"풍속({wind_speed:.1f} m/s)이 컷인(Cut-in) 속도 3 m/s 미만으로 풍력 터빈이 작동하지 않습니다."
            if wind_speed > 25.0:
                return f"풍속({wind_speed:.1f} m/s)이 컷아웃(Cut-out) 속도 25 m/s 초과로 터빈 보호를 위해 정지됩니다."
            return f"풍력 발전량이 0 MW로 억제되었습니다."

        parts = [f"풍력 발전량은 {wind_mw:.2f} MW로 예측되었습니다."]

        if wind_speed >= 12:
            parts.append(f"강한 풍속({wind_speed:.1f} m/s)으로 풍력 출력이 높습니다.")
        elif wind_speed >= 7:
            parts.append(f"풍속({wind_speed:.1f} m/s)이 적정 범위로 안정적인 발전이 예상됩니다.")
        elif wind_speed >= 3:
            parts.append(f"풍속({wind_speed:.1f} m/s)이 낮아 출력이 제한됩니다.")

        if abs(pressure_change) >= 3:
            direction = "하강" if pressure_change < 0 else "상승"
            parts.append(f"기압이 24시간 평균 대비 {abs(pressure_change):.1f} hPa {direction} 중으로 기상 변화가 감지됩니다.")

        return " ".join(parts)

    def _build_rag_comparison(self, solar_mw: float, wind_mw: float, similar_cases: list) -> str:
        if not similar_cases:
            return "유사 과거 기상 패턴이 데이터베이스에서 검색되지 않았습니다."

        best = similar_cases[0]
        rag_solar = best.get("solar_mw", 0.0)
        rag_wind = best.get("wind_mw", 0.0)
        rag_ts = best.get("timestamp", "알 수 없음")
        dist = best.get("distance", 0.0)

        solar_diff = solar_mw - rag_solar
        wind_diff = wind_mw - rag_wind
        solar_sign = "+" if solar_diff >= 0 else ""
        wind_sign = "+" if wind_diff >= 0 else ""

        similarity_label = "매우 유사" if dist < 2.0 else ("유사" if dist < 5.0 else "다소 다름")

        parts = [
            f"과거 `{rag_ts}` 기상 패턴과 {similarity_label}한 조건입니다(L2 거리: {dist:.2f}).",
            f"당시 실적: 태양광 {rag_solar:.1f} MW / 풍력 {rag_wind:.1f} MW.",
            f"이번 예측과의 차이: 태양광 {solar_sign}{solar_diff:.1f} MW / 풍력 {wind_sign}{wind_diff:.1f} MW.",
        ]

        if len(similar_cases) >= 3:
            avg_solar = np.mean([c.get("solar_mw", 0) for c in similar_cases])
            avg_wind = np.mean([c.get("wind_mw", 0) for c in similar_cases])
            parts.append(
                f"상위 {len(similar_cases)}개 유사 사례 평균: 태양광 {avg_solar:.1f} MW / 풍력 {avg_wind:.1f} MW."
            )

        return " ".join(parts)

    def _extract_kg_rules(self, kg_context: dict, window: np.ndarray) -> list:
        applied = []
        last = window[-1]
        sunshine = last[7]
        wind_speed = last[0]

        # 야간 억제
        if sunshine <= 0.01:
            applied.append("☀️ 물리 규칙 적용: 야간(일조량 ≤ 0.01) → 태양광 출력 = 0 MW")

        # 컷인/컷아웃 제약
        if wind_speed < 3.0:
            applied.append(f"💨 물리 규칙 적용: 풍속 {wind_speed:.1f} m/s < 컷인 속도 3 m/s → 풍력 = 0 MW")
        elif wind_speed > 25.0:
            applied.append(f"💨 물리 규칙 적용: 풍속 {wind_speed:.1f} m/s > 컷아웃 속도 25 m/s → 터빈 정지")

        # KG 이벤트
        for event in kg_context.get("events", []):
            severity = event.get("severity", "")
            etype = event.get("event_type", "")
            desc = event.get("description", "")
            label = f"🌪️ KG 이벤트({etype}, 심각도: {severity}): {desc}"
            applied.append(label)

        # KG 물리 규칙
        for rule in kg_context.get("rules", []):
            target = rule.get("target", "")
            condition = rule.get("condition", "")
            constraint = rule.get("constraint_value", "")
            applied.append(f"📐 KG 규칙({target}): {condition} → 제약값 {constraint}")

        return applied if applied else ["물리 규칙 및 KG 이벤트 해당 없음 (정상 운전 조건)"]

    def _explain_confidence(
        self,
        confidence: str,
        warnings: list,
        similar_cases: list,
        refined_report: Optional[dict],
    ) -> str:
        if confidence == "High":
            base = "✅ 신뢰도 높음: 기상 조건이 안정적이고 유사 사례와의 편차가 작습니다."
        elif confidence == "Medium":
            base = "⚠️ 신뢰도 중간: 기상 이변 징후가 감지되어 예측 불확실성이 다소 높습니다."
        else:
            base = "🔴 신뢰도 낮음: 급격한 기상 변화가 감지되거나 유사 사례가 부족합니다."

        parts = [base]

        if warnings:
            parts.append(f"감지된 이상 조건: {len(warnings)}건 ({', '.join(warnings[:2])}{'...' if len(warnings) > 2 else ''}).")

        if refined_report:
            low = refined_report.get("range_low", {})
            high = refined_report.get("range_high", {})
            if low and high:
                s_low = low.get("solar_mw", 0)
                s_high = high.get("solar_mw", 0)
                w_low = low.get("wind_mw", 0)
                w_high = high.get("wind_mw", 0)
                parts.append(
                    f"자율 재예측 불확실성 범위: 태양광 {s_low:.1f}~{s_high:.1f} MW / "
                    f"풍력 {w_low:.1f}~{w_high:.1f} MW."
                )

        return " ".join(parts)

    def _build_full_narrative(
        self,
        solar_mw: float,
        wind_mw: float,
        solar_explanation: str,
        wind_explanation: str,
        rag_comparison: str,
        kg_rules_applied: list,
        confidence_narrative: str,
        refined_report: Optional[dict],
    ) -> str:
        total_mw = solar_mw + wind_mw

        lines = [
            f"### 📊 종합 발전량 예측 설명",
            f"",
            f"총 예상 발전량: {total_mw:.2f} MW (태양광 {solar_mw:.2f} + 풍력 {wind_mw:.2f})",
            f"",
            f"☀️ 태양광 분석",
            solar_explanation,
            f"",
            f"💨 풍력 분석",
            wind_explanation,
            f"",
            f"📚 과거 유사 사례 비교",
            rag_comparison,
            f"",
            f"📐 물리 제약 적용 현황",
        ]

        for rule in kg_rules_applied:
            lines.append(f"- {rule}")

        lines += [
            f"",
            f"🎯 예측 신뢰도",
            confidence_narrative,
        ]

        if refined_report and refined_report.get("refinement_applied"):
            lines += [
                f"",
                f"🔁 자율 재예측 결과",
                refined_report.get("refinement_reason", "재예측이 수행되었습니다."),
            ]

        return "\n".join(lines)
