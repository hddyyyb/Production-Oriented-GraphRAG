# src/rag/answer.py

from typing import List, Dict
import numpy as np

def answer(question: str, embed_text, chunks_meta: List[Dict], llm_fn, top_k: int = 5) -> Dict[str, any]:
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
    retrieved_chunks = retrieve_top_k_chunks(query_vector, chunks_meta, top_k)
    
    # 第三步：构建上下文
    context = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])
    
    # 第四步：将问题和上下文传入LLM函数生成答案
    prompt = f"问题：{question}\n\n以下是相关的上下文信息：\n{context}\n\n请根据以上信息回答问题。"
    
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

def retrieve_top_k_chunks(query_vector: np.ndarray, chunks_meta: List[Dict], top_k: int) -> List[Dict]:
    """
    使用余弦相似度或其他相似度度量方法检索最相关的top_k片段
    这个函数模拟了检索过程
    """
    similarities = []
    for chunk in chunks_meta:
        # 计算查询向量和文档块向量之间的余弦相似度
        chunk_vector = embed_text(chunk['text'])  # 将文档块文本转化为向量
        similarity = np.dot(query_vector, chunk_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(chunk_vector))
        similarities.append((similarity, chunk))
    
    # 按照相似度排序并返回最相关的top_k片段
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_k_chunks = [chunk for _, chunk in similarities[:top_k]]
    
    return top_k_chunks
