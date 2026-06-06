import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("[DEBUG 1] Importing JejuPredictAgent...")
from app.agent import JejuPredictAgent

print("[DEBUG 2] Instantiating JejuPredictAgent...")
agent = JejuPredictAgent()

# 24시간 더미 기상 데이터 생성 (Raw scale)
dummy_window = []
for i in range(24):
    dummy_window.append([
        6.2, 0.7, 0.71, 22.1, 62.0, 1012.5, 3.0, 0.9, 1.4
    ])
dummy_window = np.array(dummy_window)

print("[DEBUG 3] Starting run_pipeline...")
try:
    print("[DEBUG 3.1] Scaling input...")
    if agent.feature_scaler is not None:
        scaled = agent.feature_scaler.transform(dummy_window)
        print("[DEBUG 3.2] Scaled successfully. Shape:", scaled.shape)
    else:
        print("[DEBUG 3.2] feature_scaler is None!")
        
    print("[DEBUG 3.3] Executing model forward...")
    if agent.model is not None:
        import torch
        x_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(agent.device)
        x_image = torch.zeros((1, 3, 112, 112), dtype=torch.float32).to(agent.device)
        with torch.no_grad():
            preds_scaled = agent.model(x_tensor, x_image).cpu().numpy()
        print("[DEBUG 3.4] Model output shape:", preds_scaled.shape)
        
        preds_orig = agent.target_scaler.inverse_transform(preds_scaled)[0]
        print("[DEBUG 3.5] Inverse scaled predictions:", preds_orig)
    else:
        print("[DEBUG 3.4] model is None!")
        
    print("[DEBUG 3.6] Executing FAISS search...")
    if agent.rag_service.index is not None:
        flat_vector = scaled.reshape(1, -1)
        # faiss 정규화 과정이 있었는지? (Colab: faiss.normalize_L2)
        # normalize_L2를 거치지 않아서 오류가 났을 수 있으나 float32인지 확인
        print("[DEBUG 3.7] Searching FAISS index...")
        similar_cases = agent.rag_service.get_similar_cases(flat_vector, top_k=3)
        print("[DEBUG 3.8] FAISS results:", similar_cases)
        
    print("[DEBUG 3.9] Executing KG query...")
    if similar_cases:
        target_ts = similar_cases[0]["timestamp"]
        kg_context = agent.kg_service.query_similar_events(target_ts)
        print("[DEBUG 3.10] KG results:", kg_context)
        
    print("[DEBUG 3.11] run_pipeline finished.")
    
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()
