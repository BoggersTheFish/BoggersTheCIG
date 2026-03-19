import argparse
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import json
import random
from collections import deque

class SimpleTSNode:
    """Proper pickleable node class"""
    def __init__(self, node_id, base_strength=1.0):
        self.id = node_id
        self.base_strength = base_strength
        self.activation = 0.0

    def score(self):
        return self.activation * self.base_strength

class TSGraph:
    def __init__(self):
        self.nodes = {}      # id -> SimpleTSNode
        self.edges = {}
        self.history = []

    def add_node(self, node_id, strength=1.0):
        if node_id not in self.nodes:
            self.nodes[node_id] = SimpleTSNode(node_id, strength)

    def connect(self, u, v, weight=0.8):
        key = tuple(sorted([u, v]))
        self.edges[key] = weight

    def propagate(self, source, intensity=1.0, decay=0.85):
        queue = deque([(source, intensity)])
        visited = set()
        while queue:
            nid, act = queue.popleft()
            if nid in visited: continue
            visited.add(nid)
            if nid in self.nodes:
                self.nodes[nid].activation += act
            for (a, b), w in list(self.edges.items()):
                neigh = b if a == nid else a if b == nid else None
                if neigh and neigh not in visited:
                    queue.append((neigh, act * w * decay))

    def relax(self, decay=0.92):
        for node in self.nodes.values():
            node.activation *= decay

    def run_cycle(self):
        if self.nodes:
            source = random.choice(list(self.nodes.keys()))
            self.propagate(source)
            self.relax()
            strongest = max(self.nodes.keys(), key=lambda k: self.nodes[k].score())
            self.nodes[strongest].base_strength += 0.05

    def get_metrics(self):
        return len(self.nodes), len(self.edges), 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--incremental', type=lambda x: x.lower() == 'true', default=True)
    args = parser.parse_args()

    SNAPSHOT = 'graph_snapshot.pkl'
    COHERENCE_LOG = 'eval/coherence_log.jsonl'
    PNG_PATH = 'obsidian/TS-Knowledge-Vault/snapshots/coherence-trend.png'

    graph = TSGraph()
    history = []

    # Load previous snapshot safely
    if args.incremental and os.path.exists(SNAPSHOT):
        try:
            with open(SNAPSHOT, 'rb') as f:
                data = pickle.load(f)
                for nid, ndata in data.get('nodes', {}).items():
                    graph.add_node(nid, ndata.get('base_strength', 1.0))
                graph.edges = data.get('edges', {})
                history = data.get('history', [])
            print("✅ Loaded previous snapshot")
        except Exception as e:
            print(f"Snapshot load failed (starting fresh): {e}")

    # Seed if empty
    if not graph.nodes:
        for i in range(12):
            graph.add_node(f"concept_{i}", 1.0 + i * 0.1)

    # Run 12 fast cycles
    for _ in range(12):
        graph.run_cycle()
        n, e, c = graph.get_metrics()
        history.append({
            "cycle": len(history),
            "nodes": n,
            "edges": e,
            "conflicts": c,
            "timestamp": datetime.now().isoformat()
        })

    # Save pickle-safe snapshot
    snapshot_data = {
        'nodes': {nid: {'base_strength': node.base_strength} for nid, node in graph.nodes.items()},
        'edges': graph.edges,
        'history': history
    }
    with open(SNAPSHOT, 'wb') as f:
        pickle.dump(snapshot_data, f)

    # Generate PNG
    cycles = [h['cycle'] for h in history]
    nodes = [h['nodes'] for h in history]
    edges = [h['edges'] for h in history]
    conflicts = [h['conflicts'] for h in history]

    plt.figure(figsize=(10, 5))
    plt.plot(cycles, nodes, label='Nodes', color='#1f77b4', linewidth=2)
    plt.plot(cycles, edges, label='Edges', color='#ff7f0e', linewidth=2)
    plt.plot(cycles, conflicts, label='Conflicts', color='#2ca02c', linewidth=2)
    plt.title('TS Coherence Trend')
    plt.xlabel('Cycle')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(PNG_PATH), exist_ok=True)
    plt.savefig(PNG_PATH, dpi=200)
    plt.close()

    print(f"✅ coherence-trend.png updated | Nodes: {nodes[-1]}")

    # Append log
    os.makedirs('eval', exist_ok=True)
    with open(COHERENCE_LOG, 'a') as f:
        for entry in history[-12:]:
            f.write(json.dumps(entry) + '\n')
