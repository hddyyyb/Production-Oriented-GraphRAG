# src/graph/build_graph.py
import json
import networkx as nx
from .extract import extract_entities, extract_relations

def build_graph(chunks_meta_path: str):
    chunks = json.load(open(chunks_meta_path, "r", encoding="utf-8"))
    G = nx.MultiDiGraph()
    for c in chunks:
        chunk_id = c["chunk_id"]
        text = c["text"]
        G.add_node(chunk_id, type="chunk", doc_id=c["doc_id"], section=c.get("section",""))
        ents = extract_entities(text)
        for e in ents:
            G.add_node(e, type="entity")
            G.add_edge(chunk_id, e, key="MENTIONS", type="MENTIONS")
        for h,r,t in extract_relations(text, ents):
            G.add_edge(h, t, key=r, type=r)
    return G
