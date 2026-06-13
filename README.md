# 📊 제주 신재생에너지 발전량 예측 및 관제 시스템

본 프로젝트는 기상 시계열 정보와 천리안 위성 구름 밀도 이미지를 융합(Fusion)하여 제주도 지역의 태양광 및 풍력 발전량을 1시간 단위로 실시간 관제 및 예측하는 AI 에이전트 시스템입니다.

## 🛠️ 핵심 기능
1. **Multi-Modal Fusion Model**: LSTM 기상 시계열 처리 브랜치와 CNN(ResNet18) 위성 이미지 특징 추출 브랜치를 융합하여 정밀한 예측 수행
2. **물리 규칙 기반 제약 (Physics-Guided)**: 일조량 부족 시 태양광 출력 0 MW 마스킹 및 컷인/컷아웃 풍속 이탈 시 풍력 출력 0 MW 마스킹 처리
3. **자율 재예측 (Self-Refinement)**: 기상 이상 감지 등으로 신뢰도가 낮아질 경우 3시간 전 슬라이딩 윈도우 앙상블을 통한 불확실성 상/하한 범위(MW) 도출
4. **설명 가능성 에이전트 (Explainability)**: 예측 원인을 기상 기여도, 유사 사례 검색(RAG), 물리 제약 위반 여부와 연결한 자연어 요약 제공
5. **멀티스텝 대화 에이전트 (Agentic Chat)**: 사용자 의도 분석 후 RAG/KG/모델 도구들을 단계적으로 오케스트레이션하여 최적의 분석 리포트 제공

---

## 📂 프로젝트 구조
- [app/main.py](file:///Users/jinho/Jeju_predict/app/main.py): FastAPI 백엔드 라우터 및 엔드포인트 정의
- [app/agent.py](file:///Users/jinho/Jeju_predict/app/agent.py): 데이터 로더 및 자율 재예측 예측 파이프라인
- [app/services/agentic_chat.py](file:///Users/jinho/Jeju_predict/app/services/agentic_chat.py): Claude API 연동 에이전트 대화 서비스
- [app/services/explainer.py](file:///Users/jinho/Jeju_predict/app/services/explainer.py): 기상 변수 영향력 및 RAG/KG 통합 설명서 생성 서비스
- [docs/PRD.md](file:///Users/jinho/Jeju_predict/docs/PRD.md): 시스템 요구사항 및 설계서 (데이터 범위: 2023.02.02 ~ 2025.12.31)
- [docs/RUN.md](file:///Users/jinho/Jeju_predict/docs/RUN.md): 로컬 가상환경 세팅 및 프로젝트 상세 실행 가이드

---

## 🚀 실행 방법
프로젝트 가상환경 설정, API 키 셋업 및 Uvicorn 서버 구동을 포함한 세부 가이드는 **[실행 가이드(docs/RUN.md)](file:///Users/jinho/Jeju_predict/docs/RUN.md)** 문서에서 확인하실 수 있습니다.
