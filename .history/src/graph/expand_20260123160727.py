# src/graph/expand.py
from typing import List, Set
import networkx as nx

def expand_entities(G: nx.MultiDiGraph, seed_chunks: List[str], hops: int = 2, max_entities: int = 30) -> Set[str]:
    seeds = set()
    for ch in seed_chunks:
        for _, e, data in G.out_edges(ch, data=True):
            if data.get("type") == "MENTIONS":
                seeds.add(e)
    # BFS扩展
    frontier = set(seeds)
    visited = set(seeds)
    for _ in range(hops-1):
        nxt = set()
        for e in frontier:
            for _, nb, _ in G.out_edges(e, data=True):
                if nb not in visited and G.nodes.get(nb, {}).get("type") == "entity":
                    nxt.add(nb)
        visited |= nxt
        frontier = nxt
        if len(visited) >= max_entities:
            break
    return set(list(visited)[:max_entities])

def chunks_for_entities(G: nx.MultiDiGraph, entities: Set[str], max_chunks: int = 30) -> List[str]:
    hits = []
    for e in entities:
        for ch, _, data in G.in_edges(e, data=True):
            if data.get("type") == "MENTIONS":
                hits.append(ch)
    # 去重保序
    seen = set()
    out = []
    for x in hits:
        if x not in seen:
            out.append(x)
            seen.add(x)
        if len(out) >= max_chunks:
            break
    return out
