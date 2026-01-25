from typing import List, Dict
import numpy as np
import re
from typing import Optional, Set
from typing import Optional
try:
    from src.graph.expand import expand_entities, chunks_for_entities
except Exception:
    expand_entities = None
    chunks_for_entities = None



def infer_allowed_doc_ids(question: str) -> Optional[Set[str]]:
    q = question.lower()
    # 最小可用：关键词门控。你后面再升级成分类器也行
    if re.search(r"\bloan(s)?\b|\blending\b|\bcredit\b|\bmortgage\b|\bund(er)?writing\b", q):
        return {"loan_policy", "credit_risk_manual", "kyc_aml_guideline"}
    return None  # None代表不做门控，兼容你其他问题

def filter_chunks_by_doc(retrieved_chunks, allowed_doc_ids: Optional[Set[str]]):
    if not allowed_doc_ids:
        return retrieved_chunks
    return [c for c in retrieved_chunks if c.get("doc_id") in allowed_doc_ids]


def trim_context_to_budget(context: str, tokenizer, question: str, instruction: str, max_ctx=2048, max_new_tokens=150):
    '''
      最大上下文长度 max_ctx(如2048 / 4096 / 8192)
      还要给模型预留生成空间(max_new_tokens)
      直接把instruction + question + context
      丢进去容易超长 → 推理报错 or 被强行截断。
    做法：
      instruction 和 question 不动
      只裁剪 context
      裁剪到“刚好能塞进模型”的长度
    '''
    buffer = 16
    budget = max_ctx - max_new_tokens - buffer  # prompt（输入部分）最多只能占这么多 token

    # 先计算 instruction+question 占多少 token
    base = f"{instruction}\n\nQuestion:\n{question}\n\nContext:\n"
    base_ids = tokenizer(base, add_special_tokens=False)["input_ids"]

    # context 可用 token
    ctx_budget = max(0, budget - len(base_ids))
    # 对 context 做 token 级截断（而不是字符串）
    ctx_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    ctx_ids = ctx_ids[:ctx_budget]  # 只截 context
    trimmed_ctx = tokenizer.decode(ctx_ids, skip_special_tokens=True)

    return trimmed_ctx



def answer(question: str, tokenizer, embed_text, chunks_meta, index, llm_fn, top_k: int = 5, G=None):
    query_vec = embed_text(question).astype("float32")
    query_vec = np.expand_dims(query_vec, 0)
    k_search = max(top_k * 3, top_k)
    distances, idxs = index.search(query_vec, k_search)

    candidates = [chunks_meta[i] for i in idxs[0]]
    allowed = infer_allowed_doc_ids(question)    # 是否包含贷款 / 信贷 / 风控相关词
    retrieved_chunks = filter_chunks_by_doc(candidates, allowed)    # 保证loan问题不会再引用tourism_guide/customer_service/ecommerce这种离谱chunk。其他类型应该是有其他处理，但是这里没写



    '''GraphRAG的“灵魂”-
    在向量召回的“局部相似性”基础上，引入“实体关系图”，
    把检索从“相似文本”升级为“语义相关证据子图”，
    用来补全流程型、跨段落的信息。'''
    # GraphRAG扩展：seed chunks -> entities -> more chunks
    chunk_lookup = {c["chunk_id"]: c for c in chunks_meta}  #  建一个 O(1) 的索引表，目的是：后面用chunk_id就能立刻拿到chunk文本

    seed_chunks = [c["chunk_id"] for c in retrieved_chunks]

    graph_chunks = []
    if G is not None and expand_entities is not None and chunks_for_entities is not None:  #  expand_entities, chunks_for_entities是import的两个函数
        ents = expand_entities(G, seed_chunks, hops=2, max_entities=30)
        more_chunk_ids = chunks_for_entities(G, ents, max_chunks=30)

        # 反查回chunk文本
        for cid in more_chunk_ids:
            if cid in chunk_lookup:
                graph_chunks.append(chunk_lookup[cid])

    # 合并：先保留seed，再追加graph扩展；去重
    merged = []
    seen = set()
    for c in (retrieved_chunks + graph_chunks):
        cid = c["chunk_id"]
        if cid not in seen:
            merged.append(c)
            seen.add(cid)

    # 再做一次domain gate过滤（避免图扩展把你拉到别的domain）
    merged = filter_chunks_by_doc(merged, allowed)

    retrieved_chunks = merged[:top_k]






    # 如果过滤后空了：直接拒答（银行场景关键能力）
    if len(retrieved_chunks) == 0:
        return {
            "answer": "I don't know based on the provided documents.",
            "citations": [],
            "graph_context": {"retrieved_chunks": [], "note": "no relevant chunks after domain gate"}
        }

    context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])

    #instruction = "Answer the question using ONLY the context. If missing, say you don't know."
    instruction = (
        "You are a banking policy assistant.\n"
        "Answer using ONLY the Evidence.\n"
        "If the Evidence is not about loan approval, say: \"I don't know based on the provided documents.\".\n"
        "Output format:\n"
        "Answer: <1-3 sentences>\n"
        "Steps:\n"
        "1.<step> (chunk_id)\n"
        "2.<step> (chunk_id)\n"
    )
    # 逼模型给“步骤+chunk_id”
    
    context_trimmed = trim_context_to_budget(context, tokenizer, question, instruction)

    # prompt = f"{instruction}\n\nContext:\n{context_trimmed}\n\nQuestion:\n{question}\n\nAnswer:"
    prompt = (
        f"{instruction}\n"
        f"Evidence:\n{context_trimmed}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n"
    )
    ans = llm_fn(prompt)

    citations = [c["chunk_id"] for c in retrieved_chunks]

    # 让graph_context里带上每个chunk的doc_id和前100字，方便面试官看到“证据集”。
    evidence_pack = [{"chunk_id": c["chunk_id"], "doc_id": c.get("doc_id", ""), "preview": c["text"][:120].replace("\n", " ")} for c in retrieved_chunks]
    citations = [c["chunk_id"] for c in retrieved_chunks]
    return {
        "answer": ans,
        "citations": citations,
        "graph_context": {
            "retrieved_chunks": citations,
            "evidence": evidence_pack,
            "domain_gate": sorted(list(allowed)) if allowed else None
        }
    }

    #return {"answer": ans, "citations": citations, "graph_context": {"retrieved_chunks": citations}}


def retrieve_top_k_chunks(query_vector: np.ndarray, chunks_meta: List[Dict], top_k: int, embed_text) -> List[Dict]:
    """
    使用FAISS索引获取最相关的top_k片段
    这个函数不再计算相似度，而是直接从FAISS检索
    """
    similarities = []
    for chunk in chunks_meta:
        chunk_vector = embed_text(chunk['text'])  # 将文档块文本转化为向量
        similarity = np.dot(query_vector, chunk_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(chunk_vector))
        similarities.append((similarity, chunk))
    
    # 按照相似度排序并返回最相关的top_k片段
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_k_chunks = [chunk for _, chunk in similarities[:top_k]]
    
    return top_k_chunks
