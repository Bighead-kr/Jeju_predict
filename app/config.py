import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Saved assets paths
MODEL_DIR = os.path.join(BASE_DIR, "models_saved")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
FAISS_INDEX_PATH = os.path.join(MODEL_DIR, "faiss_index.bin")
KG_GRAPH_PATH = os.path.join(MODEL_DIR, "kg_graph.pkl")

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
