from dotenv import load_dotenv
import os
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np

from app.agent import JejuPredictAgent
from app.services.explainer import ExplainerService
from app.services.agentic_chat import AgenticChatService

app = FastAPI(
    title="제주도 신재생에너지 예측 에이전트 시스템",
    description="LSTM 예측 모델 + KG-RAG 하이브리드 에이전트 시스템 (자율 재예측 / 멀티스텝 Q&A / 예측 설명 탑재)",
    version="2.0.0"
)

# ── 싱글톤 서비스 초기화 ────────────────────────────────────────────────────
agent = JejuPredictAgent()
explainer = ExplainerService()
agentic_chat = AgenticChatService(agent=agent, explainer=explainer)

# index.html이 위치할 templates 폴더 경로 설정
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
os.makedirs(TEMPLATES_DIR, exist_ok=True)


# ── 요청/응답 스키마 ────────────────────────────────────────────────────────
class WeatherInput(BaseModel):
    history_24h: List[List[float]] = Field(
        ...,
        description="과거 24시간 동안의 9개 기상 변수 시퀀스 (shape: 24 x 9)"
    )
    timestamp: Optional[str] = Field(None, description="예측 대상 타임스탬프")

class ChatInput(BaseModel):
    message: str = Field(..., description="사용자 질문 메시지")


# ── 라우터 ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    index_file = os.path.join(TEMPLATES_DIR, "index.html")
    if not os.path.exists(index_file):
        raise HTTPException(status_code=404, detail="Dashboard UI 파일이 생성되지 않았습니다.")
    with open(index_file, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/predict", summary="신재생에너지 발전량 예측 (자율 재예측 포함)")
async def predict_generation(input_data: WeatherInput):
    """
    #1 자율 재예측 루프 포함:
    confidence Low 시 윈도우 보정 후 재예측 → 불확실성 범위 반환
    """
    window = np.array(input_data.history_24h)

    if window.shape != (24, 9):
        raise HTTPException(
            status_code=400,
            detail=f"입력 데이터 형상은 (24, 9)여야 합니다. 현재: {window.shape}"
        )

    try:
        # 자율 재예측 루프 실행 (#1)
        report = agent.run_pipeline_with_refinement(window, timestamp=input_data.timestamp)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/explain", summary="예측 결과에 대한 자연어 설명 생성 (#4)")
async def predict_with_explanation(input_data: WeatherInput):
    """
    #4 예측 설명 에이전트:
    입력 변수 기여도 + RAG 유사사례 비교 + KG 물리 규칙 적용 내역을 자연어로 생성
    """
    window = np.array(input_data.history_24h)

    if window.shape != (24, 9):
        raise HTTPException(
            status_code=400,
            detail=f"입력 데이터 형상은 (24, 9)여야 합니다. 현재: {window.shape}"
        )

    try:
        report = agent.run_pipeline_with_refinement(window, timestamp=input_data.timestamp)
        refined = report.get("refined_report")
        explanation = explainer.explain(window, report, refined_report=refined)
        return {
            "report": report,
            "explanation": explanation,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", summary="멀티스텝 대화형 에이전트 질의응답 (#3)")
async def chat_interaction(chat_data: ChatInput):
    """
    #3 멀티스텝 Q&A 에이전트:
    질문 의도 분류 → RAG/KG/모델/설명 도구를 단계적으로 선택·실행 → 답변 조합
    """
    try:
        result = agentic_chat.chat(chat_data.message)
        return {
            "answer": result["answer"],
            "tool_trace": result["tool_trace"],
            "report": result.get("report"),
            "explanation": result.get("explanation"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", summary="서버 상태 체크")
async def health_check():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "features": ["self_refinement", "agentic_chat", "explainability"],
        "model_loaded": agent.model is not None,
        "faiss_loaded": agent.rag_service.index is not None,
        "kg_loaded": agent.kg_service.graph is not None,
    }
