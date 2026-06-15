"""
A-6: FAISS 기반 에이전트 장기 메모리
- 예측 결과·경보 이력을 벡터로 저장
- rag_service.py 패턴 재활용
- "지난 태풍 때 예측은 어땠어?" 같은 질문에 대응
"""
import os
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Any, Optional

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEMORY_DIR  = os.path.join(BASE_DIR, "data", "memory")
INDEX_PATH  = os.path.join(MEMORY_DIR, "memory_index.bin")
META_PATH   = os.path.join(MEMORY_DIR, "memory_meta.pkl")

# 메모리 벡터 차원: [solar_mw, wind_mw, confidence(0-2), trigger(0/1), n_warnings, wind_speed, cloud, sunshine]
VEC_DIM = 8


def _to_vector(record: dict) -> np.ndarray:
    preds = record.get("predictions", {})
    conf_map = {"High": 2.0, "Medium": 1.0, "Low": 0.0}
    return np.array([
        float(preds.get("solar_mw", 0.0)),
        float(preds.get("wind_mw",  0.0)),
        conf_map.get(record.get("confidence", "High"), 1.0),
        1.0 if record.get("trigger_active") else 0.0,
        float(len(record.get("warnings", []))),
        float(record.get("wind_speed", 0.0)),
        float(record.get("cloud", 5.0)),
        float(record.get("sunshine", 0.5)),
    ], dtype=np.float32)


class AgentMemory:
    def __init__(self):
        os.makedirs(MEMORY_DIR, exist_ok=True)
        self._index = None
        self._meta: list[dict] = []
        self._load()

    def _load(self):
        try:
            import faiss
            if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
                self._index = faiss.read_index(INDEX_PATH)
                with open(META_PATH, "rb") as f:
                    self._meta = pickle.load(f)
                print(f"[Memory] FAISS 인덱스 로드: {self._index.ntotal}건")
            else:
                import faiss as _f
                self._index = _f.IndexFlatL2(VEC_DIM)
                self._meta  = []
        except ImportError:
            print("[Memory] faiss 미설치 — 장기 메모리 비활성화")
        except Exception as e:
            print(f"[Memory] 로드 실패: {e}")

    def _save(self):
        try:
            import faiss
            faiss.write_index(self._index, INDEX_PATH)
            with open(META_PATH, "wb") as f:
                pickle.dump(self._meta, f)
        except Exception as e:
            print(f"[Memory] 저장 실패: {e}")

    def store(self, record: dict) -> bool:
        if self._index is None:
            return False
        try:
            vec = _to_vector(record).reshape(1, -1)
            self._index.add(vec)
            meta = {
                "timestamp":    record.get("timestamp", datetime.now().isoformat()),
                "solar_mw":     record.get("predictions", {}).get("solar_mw", 0.0),
                "wind_mw":      record.get("predictions", {}).get("wind_mw",  0.0),
                "confidence":   record.get("confidence", "High"),
                "trigger":      record.get("trigger_active", False),
                "warnings":     record.get("warnings", []),
            }
            self._meta.append(meta)
            self._save()
            return True
        except Exception as e:
            print(f"[Memory] 저장 실패: {e}")
            return False

    def search(self, record: dict, top_k: int = 3) -> list[dict]:
        if self._index is None or self._index.ntotal == 0:
            return []
        try:
            vec = _to_vector(record).reshape(1, -1)
            k   = min(top_k, self._index.ntotal)
            distances, indices = self._index.search(vec, k)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                item = dict(self._meta[idx])
                item["distance"] = float(dist)
                results.append(item)
            return results
        except Exception as e:
            print(f"[Memory] 검색 실패: {e}")
            return []

    def recent(self, n: int = 10) -> list[dict]:
        return self._meta[-n:][::-1]


# 싱글톤
_memory: Optional[AgentMemory] = None


def get_memory() -> AgentMemory:
    global _memory
    if _memory is None:
        _memory = AgentMemory()
    return _memory
