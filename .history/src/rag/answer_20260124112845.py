from typing import List, Dict
import numpy as np



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
        这正是这个函数在干的事。'''
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


def answer(question: str, tokenizer, embed_text, chunks_meta, index, llm_fn, top_k: int = 5):
    query_vec = embed_text(question).astype("float32")
    query_vec = np.expand_dims(query_vec, 0)

    distances, idxs = index.search(query_vec, top_k)
    retrieved_chunks = [chunks_meta[i] for i in idxs[0]]

    context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])

    instruction = "Answer the question using ONLY the context. If missing, say you don't know."
    context_trimmed = trim_context_to_budget(context, tokenizer, question, instruction)

    prompt = f"{instruction}\n\nContext:\n{context_trimmed}\n\nQuestion:\n{question}\n\nAnswer:"

    ans = llm_fn(prompt)

    citations = [c["chunk_id"] for c in retrieved_chunks]
    return {"answer": ans, "citations": citations, "graph_context": {"retrieved_chunks": citations}}


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
