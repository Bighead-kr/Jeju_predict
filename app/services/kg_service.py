import os
import pickle
import networkx as nx
from app.config import KG_GRAPH_PATH

class KGService:
    def __init__(self):
        self.graph = None
        self.load_graph()

    def load_graph(self):
        if os.path.exists(KG_GRAPH_PATH):
            with open(KG_GRAPH_PATH, "rb") as f:
                self.graph = pickle.load(f)
            print("NetworkX Knowledge Graph loaded successfully.")
        else:
            print(f"Warning: KG Graph file not found at {KG_GRAPH_PATH}. Please upload it after Colab creation.")

    def query_similar_events(self, timestamp, hops=1):
        """
        특정 시점(timestamp) 노드와 연결된 이벤트 및 물리 법칙 제약을 탐색하여 반환합니다.
        """
        if self.graph is None or not self.graph.has_node(timestamp):
            return {"events": [], "rules": []}

        # 1-hop 혹은 2-hop 이웃 노드 조회
        neighbors = nx.single_source_shortest_path_length(self.graph, timestamp, cutoff=hops)
        
        events = []
        rules = []
        
        for node, dist in neighbors.items():
            node_data = self.graph.nodes[node]
            node_type = node_data.get("type")
            
            if node_type == "WeatherEvent":
                events.append({
                    "event_type": node_data.get("event_type"),
                    "description": node_data.get("description"),
                    "severity": node_data.get("severity"),
                    "distance_hops": dist
                })
            elif node_type == "PhysicsRule":
                rules.append({
                    "target": node_data.get("target"),
                    "condition": node_data.get("condition"),
                    "constraint_value": node_data.get("constraint_value"),
                    "distance_hops": dist
                })
                
        return {"events": events, "rules": rules}
