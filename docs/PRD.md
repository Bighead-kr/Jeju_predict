**제주도 신재생에너지 발전량 예측**

Multi-Modal Fusion (CNN + LSTM) · KG-RAG Agent

PRD v2.0 | 2조 | 2025.06

# **0\. 프로젝트 스냅샷**

| **항목**        | **내용**                                            |
| --------------- | --------------------------------------------------- |
| 목적            | 제주도 시간 단위 태양광·풍력 발전량(MW) 동시 예측   |
| 데이터 기간     | 2023-02-02 01:00 ~ 2025-12-31 24:00 (시간 단위, 25,536행) |
| 입력 변수       | 풍속·풍향·기온·습도·기압·전운량·일조·일사 (8종 CSV) |
| 예측 타겟       | 태양광 발전량(MW) + 풍력 발전량(MW) - 동시 출력     |
| 슬라이딩 윈도우 | 과거 24시간(t-23 ~ t-1) 입력 → t 시점 예측          |
| 목표 성능       | MAE ≤ 15 MW (태양광·풍력 각각)                      |
| 개발 환경       | Google Colab (GPU) + GitHub                         |
| 추후 확장       | KG-RAG 기반 예측 설명 에이전트                      |

# **1\. 데이터 명세**

## **1-1. 입력 데이터 (전처리 완료 CSV 8종)**

| **파일명**      | **변수**        | **평균 방식**           | **결측 수**            |
| --------------- | --------------- | ----------------------- | ---------------------- |
| wind_velocity   | 풍속 (m/s)      | 산술 평균               | 0                      |
| wind_direction  | 풍향 (16방위)   | Circular mean (sin/cos) | 0                      |
| temperature     | 기온 (°C)       | 산술 평균               | 0                      |
| humidity        | 습도 (%)        | 산술 평균               | 0                      |
| pressure        | 현지기압 (hPa)  | 산술 평균               | 0                      |
| cloud_cover     | 전운량 (10분위) | 산술 평균               | 0                      |
| sunshine        | 일조 (hr)       | 산술 평균               | 13,128 (야간 → 0 처리) |
| solar_radiation | 일사 (MJ/m²)    | 산술 평균               | 13,130 (야간 → 0 처리) |

※ 제주(184)·고산(185)·서귀포(189) 3개 지점 통합 평균. 모든 파일은 '일시' 컬럼 기준 merge 가능.

## **1-2. 타겟 데이터 (별도 확보 필요 ★)**

| **발전량 실적 데이터 - 외부 확보 필요**                                   |
| ------------------------------------------------------------------------- |
| • 한국전력거래소(KPX) 제주 계통 시간별 태양광 발전량(MW)                  |
| • 한국전력거래소(KPX) 제주 계통 시간별 풍력 발전량(MW)                    |
| • 확보 경로: KPX 전력통계정보시스템 (epsis.kpx.or.kr) 또는 공공데이터포털 |
| • 필요 기간: 2023-02-02 ~ 2025-12-31 (기상 데이터와 동일 기간)            |
| • 컬럼 형식 목표: timestamp \| solar_mw \| wind_mw                        |

## **1-3. 데이터 합치기 전략**

- 8개 기상 CSV를 '일시' 기준으로 outer merge → 단일 DataFrame
- 발전량 실적 DataFrame과 timestamp 기준 inner join
- 풍향은 sin/cos 분해 후 2개 컬럼으로 분리 (wind_sin, wind_cos)
- 야간 일조·일사 NaN → 0 fillna
- 정규화: MinMaxScaler 또는 StandardScaler (변수별 독립 적용)

# **2\. 모델 아키텍처**

## **2-1. Phase 1 - LSTM 베이스라인 (Colab 1단계)**

위성 이미지 없이 수치 기상 데이터만으로 먼저 베이스라인을 만든다. 빠른 검증과 MAE 기준선 확보가 목적이다.

| **레이어**   | **설정**                | **역할**                      |
| ------------ | ----------------------- | ----------------------------- |
| 입력         | \[batch, 24, 9\]        | 24시간 × 9개 기상 변수 시퀀스 |
| LSTM Layer 1 | hidden=128, dropout=0.2 | 시간 패턴 추출                |
| LSTM Layer 2 | hidden=64, dropout=0.2  | 고수준 특징 압축              |
| Dense Output | units=2                 | 태양광 MW, 풍력 MW 동시 출력  |
| 손실함수     | MAE + MSE 혼합          | α·MAE + (1-α)·MSE, α=0.7      |
| 옵티마이저   | Adam, lr=1e-3           | ReduceLROnPlateau 스케줄러    |

## **2-2. Phase 2 - Multi-Modal Fusion (Colab 2단계)**

위성 구름 이미지(천리안 위성)를 CNN으로 처리하여 LSTM 출력과 결합한다.

| **브랜치**  | **구성**                                   | **역할**                     |
| ----------- | ------------------------------------------ | ---------------------------- |
| CNN Branch  | EfficientNet-B0 또는 ResNet18 (pretrained) | 구름 공간 특징 벡터 추출     |
| LSTM Branch | Phase 1 모델 재사용 (가중치 로드)          | 기상 수치 시간 특징          |
| Fusion      | Concatenate → Dense(64) → ReLU             | 두 브랜치 특징 결합          |
| Output      | Dense(2)                                   | 태양광·풍력 발전량 동시 예측 |

## **2-3. 학습 전략**

- **Train / Val / Test 분할**: 2023.02.02 ~ 2025.02.15 학습 (70%) | 2025.02.16 ~ 2025.07.25 검증 (15%) | 2025.07.26 ~ 2025.12.31 테스트 (15%)
- **데이터 드리프트 주의**: 2021년 HVDC 역송 정책 변경으로 학습 데이터 단순 통합 금지. 연도별 블록으로 분리 후 단계 학습 권장
- **야간 처리**: 일조=0 구간은 태양광 타겟이 0에 수렴 - 물리 마스킹으로 강제 처리
- **배치 사이즈**: Colab GPU 메모리에 따라 32~64 권장

# **3\. KG-RAG 설계 (에이전트 확장 단계)**

모델 학습 완료 후 별도 단계로 구축한다. LLM은 사용하지 않으며, 기상 조건 기반 유사 사례 검색 + 예측 근거 제공이 핵심 목적이다.

## **3-1. KG에 넣을 것 - 4가지**

| **① 기상 도메인 지식 노드**                                             |
| ----------------------------------------------------------------------- |
| • 전운량 8이상 + 일사 급감 → '전선 접근' 이벤트 (태양광 급감 선행 신호) |
| • 풍속 15m/s 이상 + 풍향 급변 → '한라산 배풍 패턴' 관계                 |
| • 기압 3hPa/3h 하강 → '저기압 접근' → 풍력 급증 예측 신호               |
| • 계절별 탁월풍: 겨울(북서풍), 여름(남서풍) - 풍력 발전량 계절 패턴     |
| • 해무 조건: 일조=0 AND 기온>22°C AND 습도>90%                          |

| **② 과거 급변 이벤트 인덱스 (RAG 검색 핵심)**                   |
| --------------------------------------------------------------- |
| • 발전량 ±30MW 이상 급변 사례 - 날짜·기상 조건·원인 태깅        |
| • 태풍 통과 이력: 태풍명, 최대 풍속, 태양광·풍력 발전량 실적    |
| • 맑음→흐림 전환 구간 (전운량 2 이하 → 8 이상 전환, 3시간 이내) |
| • 연속 고풍속 구간 (풍속 >18m/s 지속 6시간 이상) 발전량 패턴    |

| **③ 물리 법칙 규칙 (하드 제약)**              |
| --------------------------------------------- |
| • 야간(일조=0): 태양광 출력 = 0 (강제 마스킹) |
| • 풍속 < 3m/s (Cut-in 미달): 풍력 출력 ≈ 0    |
| • 풍속 > 25m/s (Cut-out): 풍력 출력 급감      |
| • 전운량 = 0~2 (맑음): 태양광 최대 출력 구간  |

| **④ 유사 기상 패턴 인덱스 (벡터 DB 연동)**                     |
| -------------------------------------------------------------- |
| • 과거 25,536개 시점의 기상 벡터 임베딩 저장                   |
| • 쿼리 시점 기상 → 코사인 유사도 Top-K 검색                    |
| • 검색된 과거 시점의 실제 발전량을 예측 보정 참조값으로 활용   |
| • 이상치 탐지: 모델 예측과 유사 사례 발전량 괴리 > 30MW → 경고 |

## **3-2. KG 스키마 (노드 / 엣지)**

| **노드 타입**    | **주요 속성**                                                                       |
| ---------------- | ----------------------------------------------------------------------------------- |
| WeatherSnapshot  | timestamp, wind_spd, wind_dir, temp, humidity, pressure, cloud, sunshine, solar_rad |
| GenerationRecord | timestamp, solar_mw, wind_mw (실측값)                                               |
| WeatherEvent     | event_type, start_ts, end_ts, severity, description                                 |
| PhysicsRule      | target(solar/wind), condition, constraint_value                                     |

| **엣지 타입**  | **방향**                           | **의미**              |
| -------------- | ---------------------------------- | --------------------- |
| SIMILAR_TO     | WeatherSnapshot → WeatherSnapshot  | 코사인 유사도 ≥ 0.85  |
| RECORDED_AT    | WeatherSnapshot → GenerationRecord | 동일 timestamp 연결   |
| TRIGGERS       | WeatherSnapshot → WeatherEvent     | 이벤트 발생 조건 충족 |
| CONSTRAINED_BY | GenerationRecord → PhysicsRule     | 물리 규칙 적용        |

## **3-3. 에이전트 동작 흐름**

- Step 1 - 현재 기상 입력 24시간 슬라이딩 윈도우 수신
- Step 2 - 모델 예측: LSTM(+CNN) → 태양광·풍력 발전량 수치 출력
- Step 3 - KG 검색 트리거 판단 (아래 조건 중 하나 해당 시)
  - 전운량 3시간 내 ±5 이상 변화 / 풍속 > 18m/s / 기압 3h 변화 > 3hPa
- Step 4 - 벡터 DB: 현재 기상 벡터 → 유사 과거 사례 Top-5 검색
- Step 5 - KG 탐색: 유사 사례의 이벤트·물리 규칙 노드 1-hop 탐색
- Step 6 - 예측값 + 검색 결과 → 에이전트 리포트 생성 (수치 기반, LLM 미사용)

# **4\. 개발용 프롬프트 (Colab 코드 생성 시 사용)**

아래 프롬프트를 Claude/GPT에 붙여넣으면 각 단계 코드 초안을 바로 얻을 수 있다.

## **4-1. 데이터 로딩 & 전처리 프롬프트**

아래 조건에 맞는 Python 코드를 작성해줘.

\[데이터\]

\- 파일: wind_velocity, wind_direction, temperature, humidity,

pressure, cloud_cover, sunshine, solar_radiation CSV 8개

\- 컬럼: 지점명 | 일시 | 측정값 (utf-8-sig 인코딩)

\- 기간: 2023-02-02 01:00 ~ 2025-12-31 24:00 (시간 단위, 25,536행)

\[요구사항\]

1\. 8개 CSV를 '일시' 기준 merge하여 단일 DataFrame 생성

2\. 풍향(16방위)을 wind_sin, wind_cos 2개 컬럼으로 분해

3\. 일조·일사 NaN → 0 fillna (야간 처리)

4\. 타겟 발전량 CSV(timestamp, solar_mw, wind_mw)와 inner join

5\. 슬라이딩 윈도우 함수: window_size=24, step=1

입력 X shape: (N, 24, 9), 타겟 y shape: (N, 2)

6\. MinMaxScaler 적용 (변수별 독립, 역변환 함수도 포함)

7\. Train/Val/Test = 시간 순서 기준 70% / 15% / 15% 분할

Google Colab 환경 기준으로 작성하고, 각 단계 주석 포함.

## **4-2. LSTM 베이스라인 모델 프롬프트**

PyTorch로 아래 스펙의 LSTM 발전량 예측 모델을 작성해줘.

\[모델 스펙\]

\- 입력: (batch, 24, 9) - 24시간 × 9개 기상 변수

\- LSTM 2층: hidden_size=128 → 64, dropout=0.2

\- 출력: (batch, 2) - 태양광 MW, 풍력 MW 동시 예측

\- 손실함수: 0.7 \* MAE + 0.3 \* MSE

\- 옵티마이저: Adam(lr=1e-3) + ReduceLROnPlateau

\- 야간 물리 마스킹: sunshine 입력이 0이면 solar 예측에 0 곱하기

\[학습 루프 요구사항\]

\- Early stopping (patience=10, monitor val_loss)

\- 에폭마다 Train MAE / Val MAE 출력

\- 최적 가중치 자동 저장: best_model.pth

\- 최종 Test MAE, RMSE, R² 평가 출력

Google Colab GPU 환경 기준, 재현성을 위한 seed 고정 포함.

## **4-3. 벡터 DB 구축 프롬프트**

FAISS를 사용해서 아래 기상 시계열 벡터 DB를 구축하는 코드를 써줘.

\[목적\]

\- 예측 시점 기상 조건과 과거 유사 구간을 코사인 유사도로 검색

\- 검색된 과거 구간의 실제 발전량을 참조값으로 활용

\[입력\]

\- DataFrame: timestamp | wind_spd | wind_sin | wind_cos | temp

| humidity | pressure | cloud | sunshine | solar_rad (9개 변수)

\- 각 시점의 24시간 윈도우를 flatten하여 벡터화 (dim=216)

\[요구사항\]

1\. 전체 학습 데이터로 FAISS IndexFlatIP 인덱스 구축

2\. 쿼리 함수: get_similar_cases(current_window, top_k=5)

반환: \[(timestamp, cosine_sim, solar_mw, wind_mw), ...\]

3\. 인덱스 저장(faiss.write_index)·로드 기능 포함

4\. Google Colab에서 실행 가능하도록 pip install faiss-cpu 포함

## **4-4. KG 구축 프롬프트 (NetworkX 경량 버전)**

NetworkX로 기상-발전량 지식 그래프를 구축하는 코드를 써줘.

\[노드 설계\]

\- WeatherSnapshot: 각 timestamp의 기상 조건 딕셔너리

\- GenerationRecord: 해당 timestamp의 solar_mw, wind_mw

\- WeatherEvent: 이벤트 타입(태풍/전선/고풍속), 시작/종료 시각

\[엣지 설계\]

\- RECORDED_AT: WeatherSnapshot → GenerationRecord (같은 timestamp)

\- TRIGGERS: WeatherSnapshot → WeatherEvent

조건: cloud_cover>=8 AND solar_rad<0.3 → 'front_approach'

wind_speed>=18 → 'high_wind'

\- SIMILAR_TO: 코사인 유사도 >=0.85인 WeatherSnapshot 쌍

\[요구사항\]

1\. 전체 DataFrame에서 그래프 자동 생성 함수

2\. 쿼리 함수: query_similar_events(snapshot_id, hops=2)

반환: 연결된 WeatherEvent 및 GenerationRecord 목록

3\. 그래프를 pickle로 저장·로드

4\. 간단한 통계 출력: 노드 수, 엣지 수, 이벤트 타입별 빈도

## **4-5. 에이전트 파이프라인 프롬프트**

Python으로 발전량 예측 에이전트 파이프라인을 작성해줘.

LLM은 사용하지 않고, 순수 수치 기반 판단 로직으로 구현해.

\[에이전트 입력\]

\- current_window: shape (24, 9)의 기상 numpy 배열

\- trained_model: 학습된 LSTM PyTorch 모델

\- faiss_index: 벡터 DB 인덱스

\- kg: NetworkX 지식 그래프

\[에이전트 동작\]

1\. 모델 예측: solar_pred, wind_pred = model(current_window)

2\. 물리 규칙 검사:

\- sunshine==0 이면 solar_pred 강제 = 0

\- wind_speed < 3 이면 wind_pred \*= 0.1

3\. 트리거 판단: cloud 3h 변화>=5 OR wind>18 OR pressure 3h 변화>3

4\. 트리거 시 FAISS 검색 → Top-5 유사 사례 발전량 조회

5\. KG 탐색 → 활성 이벤트 확인

6\. 리포트 딕셔너리 반환:

{ solar_mw, wind_mw, confidence(high/medium/low),

similar_cases, active_events, warnings }

# **5\. 추가로 해야 할 것들 - 체크리스트**

## **5-1. 데이터 확보 (최우선)**

| **★ 발전량 실적 데이터 확보 - 없으면 아무것도 못 함**                            |
| -------------------------------------------------------------------------------- |
| • KPX 전력통계정보시스템 (epsis.kpx.or.kr) → 제주 시간별 발전량 다운로드         |
| • 필요 컬럼: timestamp \| solar_mw \| wind_mw (2023.02 ~ 2025.12)                |
| • 안 되면: 공공데이터포털 (data.go.kr) '제주 태양광 풍력 발전량' 검색            |
| • 마지막 수단: 한국에너지공단 신재생에너지 보급통계 (월별 → 시간 단위 불가 주의) |

| **위성 이미지 데이터 (Phase 2용, 선택적)**                     |
| -------------------------------------------------------------- |
| • 천리안위성 2A호 구름 영상: 국가기상위성센터 (nmsc.kma.go.kr) |
| • 제주 영역 크롭 + 시간별 매칭 필요 (별도 전처리 파이프라인)   |
| • 없으면 Phase 2를 건너뛰고 수치 데이터만으로 완성 가능        |

## **5-2. 환경 세팅**

| **항목**          | **내용**                                                            | **비고**            |
| ----------------- | ------------------------------------------------------------------- | ------------------- |
| Google Colab      | GPU 런타임 설정 (T4 이상)                                           | 학습 속도 필수      |
| Google Drive 연동 | colab + drive.mount → 모델/데이터 영구 저장                         | 세션 끊겨도 안전    |
| GitHub 연동       | 학습 코드 버전 관리                                                 | 팀 협업 필수        |
| requirements.txt  | torch, pandas, numpy, scikit-learn, faiss-cpu, networkx, matplotlib | pip install 한 번에 |

## **5-3. 벡터 DB 선택**

| **옵션**     | **특징**                                     | **판단**                         |
| ------------ | -------------------------------------------- | -------------------------------- |
| FAISS (권장) | Meta 오픈소스, CPU/GPU 모두 지원, Colab 호환 | faiss-cpu pip 설치 가능, 빠름    |
| ChromaDB     | 간단한 API, 메타데이터 필터링 편리           | 대규모 데이터엔 FAISS가 빠름     |
| Pinecone     | 클라우드 관리형, 운영 편리                   | API 키 필요, 무료 플랜 용량 제한 |

→ 권장: FAISS (Colab에서 바로 쓸 수 있고 25,536행 규모에 충분)

## **5-4. KG DB 선택**

| **옵션**          | **특징**                                | **판단**                         |
| ----------------- | --------------------------------------- | -------------------------------- |
| NetworkX (권장)   | Python 내장형, Colab 바로 사용 가능     | 소규모 프로토타입 및 발표용 최적 |
| Neo4j AuraDB Free | 클라우드 무료 티어, Cypher 쿼리, 시각화 | 에이전트 고도화 시 이전 권장     |
| SQLite + 커스텀   | 관계형으로 흉내, 가장 가벼움            | 그래프 탐색 기능 약함            |

→ 권장: Phase 1~2는 NetworkX, 에이전트 완성 후 Neo4j AuraDB 이전

## **5-5. 모델 서빙 (에이전트 단계)**

- 학습된 모델 → best_model.pth 저장 후 Google Drive에 백업
- FastAPI로 /predict 엔드포인트 구성 (입력: 24h 기상 JSON → 출력: 발전량 MW)
- Colab + ngrok으로 임시 외부 접근 URL 생성 (발표·데모용)
- 에이전트 루프: 매 시간 기상 입력 → 모델 예측 → KG-RAG 검색 → 리포트 출력

## **5-6. 평가 지표**

| **지표**      | **설명**                | **기준**              | **비고**         |
| ------------- | ----------------------- | --------------------- | ---------------- |
| MAE (주지표)  | Mean Absolute Error     | 목표: ≤ 15 MW         | 태양광·풍력 각각 |
| RMSE          | Root Mean Squared Error | 급변 구간 페널티 강화 | 참고 지표        |
| R²            | 결정계수                | 설명력 평가           | 0.85 이상 목표   |
| 야간 제외 MAE | 일조>0 구간만 평가      | 태양광 낮 성능 집중   | 추가 평가        |

# **6\. 전체 로드맵**

| **Phase** | **기간** | **주요 작업**                                         |
| --------- | -------- | ----------------------------------------------------- |
| Phase 0   | 1주차    | 발전량 실적 데이터 확보 + 8개 CSV 병합 + EDA          |
| Phase 1   | 2~3주차  | LSTM 베이스라인 학습 (수치 데이터) + MAE 기준선 확인  |
| Phase 2   | 4~5주차  | FAISS 벡터 DB 구축 + NetworkX KG 구축                 |
| Phase 3   | 6~7주차  | CNN 위성 이미지 브랜치 추가 + Multi-Modal Fusion 학습 |
| Phase 4   | 8~9주차  | 에이전트 파이프라인 통합 + 물리 마스킹 + 트리거 로직  |
| Phase 5   | 10주차   | FastAPI 서빙 + 최종 평가 + 발표 데모                  |

_PRD v2.0 | LLM 미사용 순수 수치 예측 아키텍처 | Colab 개발 기준_