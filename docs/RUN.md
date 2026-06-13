# 🚀 제주 신재생에너지 예측 시스템 실행 가이드

이 문서는 **제주도 신재생에너지 발전량 예측 및 관제 플랫폼(v2.0)**을 로컬 개발 환경에서 설정하고 실행하는 방법을 안내합니다. 본 프로젝트는 수치 기상 데이터와 위성 구름 이미지를 융합한 LSTM+CNN 멀티모달 모델, FAISS RAG, NetworkX 지식 그래프(KG) 기반의 AI 에이전트 시스템입니다.

---

## 📌 사전 요구사항

- **Python 버전**: `Python 3.12` 권장 (최소 3.10 이상)
- **주요 파일 구성 확인**:
  - 데이터셋: [jeju_solar_wind_generation.csv](file:///Users/jinho/Jeju_predict/docs/jeju_solar_wind_generation.csv) (데이터 기간: 2023.02.02 ~ 2025.12.31, 총 25,536행)
  - 학습 완료 모델 가중치: `models_saved/best_model.pth`
  - RAG 데이터 인덱스: `models_saved/faiss_index.bin` 및 `models_saved/faiss_meta.pkl`
  - 지식 그래프 파일: `models_saved/kg_graph.pkl`

---

## 🛠️ 1. 개발 환경 구축

### 1-1. 가상환경 생성 및 활성화
의존성 충돌을 방지하기 위해 가상환경을 사용하여 실행하는 것을 권장합니다.

```bash
# 가상환경 생성 (.venv)
python3 -m venv .venv

# 가상환경 활성화 (macOS/Linux)
source .venv/bin/activate
```

### 1-2. 필수 패키지 설치
[requirements.txt](file:///Users/jinho/Jeju_predict/requirements.txt)에 명시된 패키지들을 가상환경 내에 설치합니다.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> [!IMPORTANT]
> macOS 환경에서 `faiss-cpu`가 스레드 충돌을 일으키는 것을 방지하기 위해, RAG 서비스 실행 시 내부적으로 `faiss.omp_set_num_threads(1)` 조치가 자동으로 적용되어 있습니다.

### 1-3. 환경 변수 설정 (.env)
멀티스텝 대화형 에이전트(Agentic Chat) 작동을 위해 Anthropic API 키가 필요합니다. 프로젝트 루트에 [.env](file:///Users/jinho/Jeju_predict/.env) 파일을 생성하고 API 키를 설정합니다.

```ini
ANTHROPIC_API_KEY=your_claude_api_key_here
```
*(현재 로컬 .env 파일에는 테스트용 API 키가 기본 세팅되어 있습니다.)*

---

## 🏃 2. 백엔드 및 대시보드 실행

### 2-1. uvicorn 개발 서버 실행
FastAPI 백엔드 애플리케이션인 [app/main.py](file:///Users/jinho/Jeju_predict/app/main.py)를 구동합니다. 로컬 임포트 경로가 정상적으로 동작하도록 `PYTHONPATH` 환경변수를 지정하여 실행해야 합니다.

```bash
PYTHONPATH=. python3 -m uvicorn app.main:app --reload
```

- `--reload`: 소스 코드 수정 시 개발 서버가 자동으로 재기동됩니다.
- 정상적으로 실행되면 다음과 같은 로그가 콘솔에 출력됩니다:
  ```text
  FAISS Index and metadata loaded successfully from: /Users/jinho/Jeju_predict/models_saved/faiss_meta.pkl
  NetworkX Knowledge Graph loaded successfully.
  PyTorch MultiModal Prediction Model loaded successfully.
  MinMaxScaler objects loaded successfully.
  CSV 로드: 25536행 (2023-02-02 ~ 2025-12-31)
  INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
  ```

---

## 💻 3. 웹 관제 대시보드 접속 및 테스트

서버가 구동되면 브라우저를 열고 다음 URL에 접속하여 통합 관제 화면을 이용할 수 있습니다.

* **통합 관제 웹 대시보드 UI**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
* **API 대화형 문서 (Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 대시보드 주요 기능 테스트 방법
1. **정시 관제 시각 설정**: 캘린더나 시간 슬라이더를 통해 날짜 및 시간을 변경하면 실시간 기상 데이터와 해당 시점의 모델 예측 결과가 우측 패널에 즉시 반영됩니다.
   - *주의: 데이터 셋의 최종 한계 시점은 **`2025년 12월 31일 24:00`**이므로, 이 범위 내의 날짜를 선택해야 실제 실적 및 예측 데이터가 정상 로드됩니다.*
2. **AI 대화형 에이전트 (인라인 챗봇)**: 우측 하단의 챗봇 대화창에 다음과 같은 질문을 입력하여 멀티스텝 Q&A 성능을 확인합니다.
   - `"오늘 태양광 예측 발전량은 얼마야?"`
   - `"2025년 10월 24일 14시 실적이랑 예측 비교해줘"`
   - `"내일 발전량이 급감하는 원인을 설명해줘"`
3. **자율 재예측(Self-Refinement) 확인**: 기상 이변(태풍, 한라산 배풍, 강풍 경보 등)이 감지되어 신뢰도가 `Low`로 나타나는 경우, 시스템이 자동으로 3시간 전 슬라이딩 윈도우 예측 모델과 앙상블한 **재예측 범위(Min/Max MW)**가 대시보드 상단 경보판에 실시간 갱신(🔁 마크)되는지 확인합니다.

---

## 🗂️ 4. 주요 소스코드 가이드

- [app/main.py](file:///Users/jinho/Jeju_predict/app/main.py): FastAPI 라우터 및 엔드포인트 정의 (`/predict`, `/predict/explain`, `/chat`, `/health`)
- [app/agent.py](file:///Users/jinho/Jeju_predict/app/agent.py): 데이터 로딩, 전처리, LSTM+CNN 융합 예측 파이프라인 및 자율 재예측 루프 코어
- [app/services/agentic_chat.py](file:///Users/jinho/Jeju_predict/app/services/agentic_chat.py): Claude API 기반 사용자 의도 분류 및 멀티스텝 도구(RAG, KG, Predict 등) 오케스트레이션 서비스
- [app/services/explainer.py](file:///Users/jinho/Jeju_predict/app/services/explainer.py): SHAP/z-score 변수 중요도 분석, RAG 유사 구간 비교, 물리 법칙 적용 내역을 조합한 설명서 생성기
