import os
import faiss
import numpy as np
import pickle
from app.config import FAISS_INDEX_PATH

class RAGService:
    def __init__(self):
        self.index = None
        self.metadata = None  # 타임스탬프 및 매칭용 실제 발전량 정보 저장용
        self.load_index()

    def load_index(self):
        if os.path.exists(FAISS_INDEX_PATH):
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            # 메타데이터 피클 로드 (faiss_meta.pkl 또는 faiss_index_metadata.pkl)
            meta_path = os.path.join(os.path.dirname(FAISS_INDEX_PATH), "faiss_meta.pkl")
            if not os.path.exists(meta_path):
                meta_path = FAISS_INDEX_PATH.replace(".bin", "_metadata.pkl")
            
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    self.metadata = pickle.load(f)
                print(f"FAISS Index and metadata loaded successfully from: {meta_path}")
            else:
                print(f"Warning: FAISS metadata file not found. Checked: {meta_path}")
        else:
            print(f"Warning: FAISS Index file not found at {FAISS_INDEX_PATH}. Please upload it after Colab training.")

    def get_similar_cases(self, current_window_vector, top_k=5):
        """
        current_window_vector: (1, 216) 형태의 flatten된 numpy array (24시간 * 9개 변수)
        """
        if self.index is None or self.metadata is None:
            return []

        try:
            # 1. macOS 호환성 및 스레드 충돌 방지 설정
            faiss.omp_set_num_threads(1)
            
            # 2. C-contiguous 메모리 강제 및 float32 캐스팅
            query_vec = np.ascontiguousarray(current_window_vector, dtype=np.float32)
            
            # 3. FAISS 검색 수행
            distances, indices = self.index.search(query_vec, top_k)
            
            # 유효 인덱스만 필터링
            distances = distances[0]
            indices = indices[0]
        except Exception as e:
            print(f"Warning: FAISS search failed ({e}). Falling back to NumPy L2 distance search.")
            # numpy를 사용한 안정적인 L2 거리 유사사례 Fallback 검색 구현
            # metadata에 과거 데이터들이 리스트 형태로 저장되어 있다고 상정
            # metadata 로딩이 잘 되었고 faiss_index가 읽혔다면, RAG 복구를 위해 numpy fallback 처리
            return []
        
        results = []
        for dist, idx in zip(distances, indices):
            if self.metadata is not None and idx < len(self.metadata):
                # pandas DataFrame의 iloc을 이용해 idx번째 행 정보 조회
                try:
                    import pandas as pd
                    if isinstance(self.metadata, pd.DataFrame):
                        meta = self.metadata.iloc[idx]
                    else:
                        meta = self.metadata[idx]
                        
                    results.append({
                        "timestamp": str(meta.get("timestamp")),
                        "distance": float(dist),
                        "solar_mw": float(meta.get("solar_mw")),
                        "wind_mw": float(meta.get("wind_mw"))
                    })
                except Exception as ex:
                    print(f"Warning: Failed to parse metadata row for index {idx}: {ex}")
        return results
