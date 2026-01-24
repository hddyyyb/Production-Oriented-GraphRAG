import faiss
import os
import numpy as np
import json
from sentence_transformers import SentenceTransformer

def build_faiss(embeddings, chunks_meta, index_dir='data/index'):
    """
    使用 FAISS 构建并保存索引
    :param embeddings: 文本块的嵌入（向量）
    :param chunks_meta: 文本块的元数据，用于后续引用
    :param index_dir: 保存索引的目录
    """
    # 确保索引文件夹存在
    os.makedirs(index_dir, exist_ok=True)
    
    # 将嵌入转换为 numpy 数组
    embeddings = np.array(embeddings, dtype="float32")
    
    # 创建 FAISS 索引，使用 L2 距离度量
    dim = embeddings.shape[1]  # 嵌入向量的维度
    index = faiss.IndexFlatL2(dim)
    
    # 将嵌入向量添加到索引中
    index.add(embeddings)
    
    # 保存索引到文件
    faiss.write_index(index, os.path.join(index_dir, 'chunks.faiss'))
    
    # 保存元数据（用于以后检索）
    with open(os.path.join(index_dir, 'chunks_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(chunks_meta, f, ensure_ascii=False)

    print(f"FAISS 索引已成功构建并保存在 {index_dir} 中！")
