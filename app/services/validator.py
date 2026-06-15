"""
A-2: tool_validate — 물리 규칙 기반 예측값 자기검증
검증 항목:
  1. 야간 태양광 = 0
  2. cut-in(3 m/s) / cut-out(25 m/s) 풍속 위반
  3. z-score 이상치 (±3σ 초과)
  4. 설비 용량 상한 초과 (태양광 200 MW, 풍력 300 MW — 제주 기준)
"""
import numpy as np
from typing import Any

SOLAR_CAPACITY_MW = 200.0
WIND_CAPACITY_MW  = 300.0

# 통계 기반 z-score 한계 — 이 값은 학습 데이터 분포에서 추정
SOLAR_MEAN, SOLAR_STD = 60.0, 45.0
WIND_MEAN,  WIND_STD  = 80.0, 55.0
Z_THRESHOLD = 3.0


def validate_predictions(
    solar_mw: float,
    wind_mw: float,
    current_window: np.ndarray,
) -> dict[str, Any]:
    """
    예측값과 기상 윈도우를 받아 물리 규칙 검증 결과 반환.

    Returns:
        {
          "passed": bool,
          "violations": list[str],
          "corrected": {"solar_mw": float, "wind_mw": float},
        }
    """
    violations: list[str] = []
    corrected_solar = solar_mw
    corrected_wind  = wind_mw

    # --- 1. 야간 태양광 제약 ---
    sunshine_val = float(current_window[-1, 7]) if current_window.shape[1] > 7 else 1.0
    if sunshine_val <= 0.01 and solar_mw > 0.0:
        violations.append(
            f"야간 태양광 위반: 일조량 {sunshine_val:.3f} ≤ 0.01 인데 태양광 예측 {solar_mw:.2f} MW"
        )
        corrected_solar = 0.0

    # --- 2. 풍속 cut-in / cut-out 제약 ---
    wind_speed_val = float(current_window[-1, 0])
    if wind_speed_val < 3.0 and wind_mw > 0.0:
        violations.append(
            f"Cut-in 미달 위반: 풍속 {wind_speed_val:.1f} m/s < 3 m/s 인데 풍력 예측 {wind_mw:.2f} MW"
        )
        corrected_wind = 0.0
    elif wind_speed_val > 25.0 and wind_mw > 0.0:
        violations.append(
            f"Cut-out 초과 위반: 풍속 {wind_speed_val:.1f} m/s > 25 m/s 인데 풍력 예측 {wind_mw:.2f} MW"
        )
        corrected_wind = 0.0

    # --- 3. z-score 이상치 ---
    solar_z = (solar_mw - SOLAR_MEAN) / SOLAR_STD if SOLAR_STD > 0 else 0.0
    wind_z  = (wind_mw  - WIND_MEAN)  / WIND_STD  if WIND_STD  > 0 else 0.0

    if abs(solar_z) > Z_THRESHOLD:
        violations.append(
            f"태양광 z-score 이상치: {solar_mw:.2f} MW (z={solar_z:.1f})"
        )
    if abs(wind_z) > Z_THRESHOLD:
        violations.append(
            f"풍력 z-score 이상치: {wind_mw:.2f} MW (z={wind_z:.1f})"
        )

    # --- 4. 설비 용량 상한 ---
    if corrected_solar > SOLAR_CAPACITY_MW:
        violations.append(
            f"태양광 용량 상한 초과: {corrected_solar:.2f} MW > {SOLAR_CAPACITY_MW} MW"
        )
        corrected_solar = SOLAR_CAPACITY_MW
    if corrected_wind > WIND_CAPACITY_MW:
        violations.append(
            f"풍력 용량 상한 초과: {corrected_wind:.2f} MW > {WIND_CAPACITY_MW} MW"
        )
        corrected_wind = WIND_CAPACITY_MW

    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "corrected": {
            "solar_mw": max(0.0, round(corrected_solar, 2)),
            "wind_mw":  max(0.0, round(corrected_wind,  2)),
        },
    }
