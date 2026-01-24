# src/rag/answer.py
# In `src/rag/answer.py`, you can create an `answer` function to handle the question-answering requests, use FAISS to retrieve the most similar snippets, and then generate the answer using an LLM.

from typing import Dict, Any, List
from ..graph.expand import expand_entities, chunks_for_entities

def answer(question: str, vec_search_fn, chunks_meta: List[dict], G, llm_fn, topk: int = 6) -> Dict[str, Any]:
    seed_chunk_ids = vec_search_fn(question, topk=topk)  # 返回chunk_id列表
    ents = expand_entities(G, seed_chunk_ids, hops=2)
    graph_chunk_ids = chunks_for_entities(G, ents, max_chunks=20)

    # 合并上下文(去重)
    context_ids = []
    seen = set()
    for cid in seed_chunk_ids + graph_chunk_ids:
        if cid not in seen:
            context_ids.append(cid); seen.add(cid)

    id2chunk = {c["chunk_id"]: c for c in chunks_meta}
    context = "\n\n".join([f"[{cid}] {id2chunk[cid]['text']}" for cid in context_ids if cid in id2chunk])

    prompt = f"""你是银行内部知识助手。请只基于给定证据回答，答案要精确、可执行，并在每条关键结论后标注引用编号。
问题：{question}

证据：
{context}

输出格式：
- Answer: ...
- Citations: [chunk_id...]
"""
    llm_out = llm_fn(prompt)

    return {
      "answer": llm_out,
      "citations": context_ids,
      "graph_context": {
        "seed_chunks": seed_chunk_ids,
        "entities": list(ents),
        "expanded_chunks": graph_chunk_ids
      }
    }
