import numpy as np
import torch
import os
import pickle
from PIL import Image
from torchvision import transforms
from app.models.lstm_model import MultiModalFusionModel
from app.services.rag_service import RAGService
from app.services.kg_service import KGService
from app.config import MODEL_PATH, BASE_DIR

class JejuPredictAgent:
    def __init__(self):
        self.rag_service = RAGService()
        self.kg_service = KGService()
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.load_model()
        self.load_scalers()
        
        # 이미지 변환 파이프라인 (ResNet18 입력 조건과 동일하게 전처리)
        self.img_transforms = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_model(self):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = MultiModalFusionModel()
            if torch.cuda.is_available():
                self.model.load_state_dict(torch.load(MODEL_PATH))
            else:
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            self.model.to(self.device)
            self.model.eval()
            print("PyTorch MultiModal Prediction Model loaded successfully.")
        except Exception as e:
            print(f"Warning: Model could not be loaded: {e}. Please upload 'best_model.pth' to models_saved/")

    def load_scalers(self):
        try:
            model_dir = os.path.join(BASE_DIR, "models_saved")
            feature_scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
            target_scaler_path = os.path.join(model_dir, "target_scaler.pkl")
            
            if os.path.exists(feature_scaler_path):
                with open(feature_scaler_path, "rb") as f:
                    self.feature_scaler = pickle.load(f)
                # numpy array 입력 시 발생하는 feature-name 불일치 경고 방지
                if hasattr(self.feature_scaler, "feature_names_in_"):
                    del self.feature_scaler.feature_names_in_
            if os.path.exists(target_scaler_path):
                with open(target_scaler_path, "rb") as f:
                    self.target_scaler = pickle.load(f)
                if hasattr(self.target_scaler, "feature_names_in_"):
                    del self.target_scaler.feature_names_in_
            print("MinMaxScaler objects loaded successfully.")
        except Exception as e:
            print(f"Warning: Scalers could not be loaded: {e}")

    def run_pipeline(self, current_window, timestamp=None):
        """
        current_window: (24, 9) 차원의 기상 데이터 시퀀스 (Numpy Array, Raw Scale)
        timestamp: 현재 예측하려는 시점의 타임스탬프 (Optional)
        """
        # 1. 모델 예측
        solar_pred = 0.0
        wind_pred = 0.0
        
        # 기상 변수 인덱스 가이드 (Raw 값 기준):
        # 0: wind_speed, 7: sunshine
        sunshine_val = current_window[-1, 7]
        wind_speed_val = current_window[-1, 0]
        
        # 모델 입력 스케일링 수행
        current_window_scaled = current_window.copy()
        if self.feature_scaler is not None:
            current_window_scaled = self.feature_scaler.transform(current_window)
        
        if self.model is not None and self.target_scaler is not None:
            # 윈도우 차원 추가 (1, 24, 9)
            x_tensor = torch.tensor(current_window_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # 타임스탬프에 매칭되는 위성 이미지 탐색 및 로딩
            x_image = None
            if timestamp is not None:
                try:
                    from datetime import datetime
                    dt = datetime.strptime(timestamp.strip(), "%Y-%m-%d %H:%M:%S")
                    ts_str = dt.strftime("%Y%m%d_%H%M")
                    
                    satellite_dir = os.path.join(BASE_DIR, "data", "satellite", "jeju_dataset")
                    if not os.path.exists(satellite_dir):
                        satellite_dir = os.path.join(BASE_DIR, "data", "satellite")
                        
                    img_path = os.path.join(satellite_dir, f"{ts_str}.jpg")
                    
                    if os.path.exists(img_path):
                        img = Image.open(img_path).convert('RGB')
                        
                        # ★ 제주도 좌표 영역 크롭 박스 (left, top, right, bottom)
                        crop_box = (237, 301, 275, 323)
                        cropped_img = img.crop(crop_box)
                        
                        # 크롭된 이미지를 CNN 규격으로 변환하여 로드
                        x_image = self.img_transforms(cropped_img).unsqueeze(0).to(self.device)
                        print(f"Loaded matching cropped satellite image for prediction: {img_path}")
                except Exception as ex:
                    print(f"Warning: Failed to load/preprocess satellite image for {timestamp}: {ex}")
            
            if x_image is None:
                x_image = torch.zeros((1, 3, 112, 112), dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                preds_scaled = self.model(x_tensor, x_image).cpu().numpy() # shape: (1, 2)
                
            # 예측치 역스케일링 수행 (0~1 범위를 실제 발전량 MW 단위로 복원)
            preds_orig = self.target_scaler.inverse_transform(preds_scaled)[0]
            solar_pred = float(preds_orig[0])
            wind_pred = float(preds_orig[1])
        
        # 2. 하드웨어 물리 규칙 제약 적용 (PRD 3-1 ③ 참조)
        # 야간 제약: 일조량이 0.01 이하인 경우 태양광 출력 = 0
        if sunshine_val <= 0.01:
            solar_pred = 0.0
            
        # 풍속 제약: 풍속이 3m/s 미만(Cut-in) 또는 25m/s 초과(Cut-out) 이탈 시 풍력 발전량 ≈ 0
        if wind_speed_val < 3.0 or wind_speed_val > 25.0:
            wind_pred = 0.0

        # 3. 트리거 판단 및 RAG/KG 검색 실행
        # 트리거 조건: 전운량(index 6) 3시간 내 변화폭 >= 5 OR 풍속 > 18m/s OR 기압(index 5) 3시간 내 변화 > 3hPa
        trigger_active = False
        warnings = []
        
        if len(current_window) >= 3:
            # 6: cloud, 5: pressure
            cloud_diff = abs(current_window[-1, 6] - current_window[-3, 6])
            pressure_diff = abs(current_window[-1, 5] - current_window[-3, 5])
            
            if cloud_diff >= 5.0:
                trigger_active = True
                warnings.append("최근 3시간 내 전운량 급변 탐지 (전선 접근 가능성)")
            if wind_speed_val > 18.0:
                trigger_active = True
                warnings.append("고풍속 경보 (풍속 18m/s 초과)")
            if pressure_diff >= 3.0:
                trigger_active = True
                warnings.append("기압 급변 탐지 (저기압/태풍 접근 가능성)")

        # 4. RAG 및 KG 검색
        similar_cases = []
        kg_context = {"events": [], "rules": []}
        
        if trigger_active or timestamp is not None:
            # FAISS 검색을 위해 스케일링된 flat 벡터(1, 216) 생성
            flat_vector = current_window_scaled.reshape(1, -1)
            similar_cases = self.rag_service.get_similar_cases(flat_vector, top_k=3)
            
            if similar_cases:
                target_ts = similar_cases[0]["timestamp"]
                kg_context = self.kg_service.query_similar_events(target_ts)

        # 5. 종합 신뢰도 판정
        confidence = "High"
        if len(warnings) > 0:
            confidence = "Medium"
        if len(warnings) >= 2 or (similar_cases and similar_cases[0]["distance"] > 10.0):
            confidence = "Low"

        # 6. 최종 에이전트 분석 리포트 생성 (LLM 미사용)
        report = {
            "predictions": {
                "solar_mw": max(0.0, solar_pred),
                "wind_mw": max(0.0, wind_pred)
            },
            "confidence": confidence,
            "warnings": warnings,
            "trigger_active": trigger_active,
            "similar_cases": similar_cases,
            "kg_context": kg_context
        }
        
        return report

    # ── #1 자율 재예측 루프 (Self-Refinement) ────────────────────────────────

    def run_pipeline_with_refinement(self, current_window: np.ndarray, timestamp=None) -> dict:
        """
        기본 파이프라인 실행 후 신뢰도(confidence)가 'Low'인 경우:
          1) 윈도우를 3시간 전 슬라이딩으로 재예측
          2) 두 예측값을 앙상블하여 불확실성 범위를 산출
          3) refined_report 키를 결과에 포함하여 반환
        """
        # 1차 예측
        report = self.run_pipeline(current_window, timestamp=timestamp)

        # 신뢰도 High/Medium이면 재예측 불필요
        if report["confidence"] in ("High", "Medium"):
            report["refined_report"] = None
            return report

        # ── 재예측 시도 ──────────────────────────────────────────────────────
        refinement_reason_parts = []
        refinement_reason_parts.append(
            f"신뢰도 Low 감지 (경고 {len(report['warnings'])}건). "
            "3시간 전 윈도우로 자율 재예측을 수행합니다."
        )

        # 3시간 전 기상 데이터로 재구성 (윈도우를 3 타임스텝 뒤로 밀고 앞은 복제)
        window_shifted = np.roll(current_window, shift=3, axis=0)
        window_shifted[:3] = current_window[0]  # 앞부분 경계값으로 채움

        report_shifted = self.run_pipeline(window_shifted, timestamp=timestamp)

        # ── 앙상블 및 불확실성 범위 계산 ───────────────────────────────────
        solar_vals = sorted([
            report["predictions"]["solar_mw"],
            report_shifted["predictions"]["solar_mw"],
        ])
        wind_vals = sorted([
            report["predictions"]["wind_mw"],
            report_shifted["predictions"]["wind_mw"],
        ])

        # 앙상블 평균을 주 예측값으로 갱신
        ensemble_solar = float(np.mean(solar_vals))
        ensemble_wind = float(np.mean(wind_vals))
        report["predictions"]["solar_mw"] = round(ensemble_solar, 2)
        report["predictions"]["wind_mw"] = round(ensemble_wind, 2)

        refined_report = {
            "refinement_applied": True,
            "refinement_reason": " ".join(refinement_reason_parts),
            "original_predictions": {
                "solar_mw": solar_vals[0] if solar_vals[0] < solar_vals[1] else solar_vals[1],
                "wind_mw": wind_vals[0] if wind_vals[0] < wind_vals[1] else wind_vals[1],
            },
            "shifted_predictions": {
                "solar_mw": report_shifted["predictions"]["solar_mw"],
                "wind_mw": report_shifted["predictions"]["wind_mw"],
            },
            "range_low": {
                "solar_mw": round(solar_vals[0], 2),
                "wind_mw": round(wind_vals[0], 2),
            },
            "range_high": {
                "solar_mw": round(solar_vals[1], 2),
                "wind_mw": round(wind_vals[1], 2),
            },
            "ensemble_solar_mw": round(ensemble_solar, 2),
            "ensemble_wind_mw": round(ensemble_wind, 2),
        }

        report["refined_report"] = refined_report
        return report
