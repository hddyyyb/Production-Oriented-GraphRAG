from typing import List, Dict
import numpy as np



def trim_context_to_budget(context: str, question: str, instruction: str, max_ctx=2048, max_new_tokens=150):
    buffer = 16
    budget = max_ctx - max_new_tokens - buffer

    # 先计算 instruction+question 占多少 token
    base = f"{instruction}\n\nQuestion:\n{question}\n\nContext:\n"
    base_ids = tokenizer(base, add_special_tokens=False)["input_ids"]

    # context 可用 token
    ctx_budget = max(0, budget - len(base_ids))

    ctx_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    ctx_ids = ctx_ids[:ctx_budget]  # 只截 context
    trimmed_ctx = tokenizer.decode(ctx_ids, skip_special_tokens=True)

    return trimmed_ctx



def answer(question: str, tokenizer, embed_text, chunks_meta: List[Dict], llm_fn, top_k: int = 5) -> Dict[str, any]:
    """
    根据问题和检索到的文档块生成答案
    :param question: 用户提出的问题
    :param embed_text: 用于将文本转化为向量的嵌入函数
    :param chunks_meta: 所有文档块的元数据
    :param llm_fn: 用于生成答案的LLM函数
    :param top_k: 检索最相关的文档块数目
    :return: 包含答案、引用和上下文的字典
    """
    # 第一步：向量化用户的问题
    query_vector = embed_text(question)  # 将问题转为向量
    
    # 第二步：基于FAISS或其他方式检索最相关的文档块
    # 这里假设已经有了检索到的top_k个相关片段（实际上是通过FAISS索引检索）
    retrieved_chunks = retrieve_top_k_chunks(query_vector, chunks_meta, top_k, embed_text)
    
    # 第三步：构建上下文
    context = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])
    
    # 第四步：将问题和上下文传入LLM函数生成答案
    instruction = "Answer the question using ONLY the context. If missing, say you don't know."
    context_trimmed = trim_context_to_budget(context, question, instruction)
    prompt = f"{instruction}\n\nContext:\n{context_trimmed}\n\nQuestion:\n{question}\n\nAnswer:"

    # 调用LLM生成答案
    answer = llm_fn(prompt)  # LLM生成的答案
    
    # 第五步：返回结果（答案 + 引用 + 上下文）
    citations = [chunk['chunk_id'] for chunk in retrieved_chunks]
    graph_context = {
        "retrieved_chunks": [chunk['chunk_id'] for chunk in retrieved_chunks],
    }
    
    return {
        "answer": answer,
        "citations": citations,
        "graph_context": graph_context
    }

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
