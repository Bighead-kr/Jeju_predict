import time
import numpy as np
from app.agent import JejuPredictAgent
from app.services.explainer import ExplainerService
from app.services.agentic_chat import AgenticChatService
from app.services.kma_service import KMAService

def benchmark_daily_data_simulation():
    print("=== 1. daily_data 예측 루프 벤치마크 ===")
    agent = JejuPredictAgent()
    explainer = ExplainerService()
    agentic_chat = AgenticChatService(agent=agent, explainer=explainer)
    
    date = "2026-06-15"
    
    # 24시간 루프 예측 시뮬레이션
    start_time = time.time()
    
    solar_sum = 0.0
    wind_sum = 0.0
    hourly = []
    
    for h in range(24):
        ts = f"{date} {h:02d}:00:00"
        ts_start = time.time()
        
        # 윈도우 생성
        window = agentic_chat._build_window(ts)
        
        # RAG/KG 생략을 적용한 예측
        report = agent.run_pipeline_with_refinement(window, timestamp=ts, skip_rag_kg=True)
        
        solar = report["predictions"]["solar_mw"]
        wind = report["predictions"]["wind_mw"]
        solar_sum += solar
        wind_sum += wind
        
        hourly.append({
            "hour": f"{h:02d}",
            "solar_mw": solar,
            "wind_mw": wind,
        })
        
        # print(f"  {h:02d}시 완료: {time.time() - ts_start:.4f}초")
        
    elapsed = time.time() - start_time
    print(f"24시간 전체 예측 시뮬레이션 소요시간: {elapsed:.4f}초")
    print(f"총 발전량 예측 합계 - 태양광: {solar_sum:.2f} MW, 풍력: {wind_sum:.2f} MW")
    
    assert elapsed < 5.0, "성능 최적화 실패: 24시간 예측 소요시간이 5초 이상입니다!"
    print("✅ 예측 루프 성능 테스트 통과!")

if __name__ == "__main__":
    benchmark_daily_data_simulation()
