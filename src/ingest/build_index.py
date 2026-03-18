import faiss
import os
import numpy as np
import json

def build_faiss(embeddings, chunks_meta, index_dir='data/index'):
    """
    Build and save a FAISS index
    :param embeddings: Embeddings (vectors) of text chunks
    :param chunks_meta: Metadata of text chunks for later retrieval
    :param index_dir: Directory to save the index
    """
    os.makedirs(index_dir, exist_ok=True)
    
    embeddings = np.array(embeddings, dtype="float32")
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)    # Create FAISS index using L2 distance metric
    
    index.add(embeddings)    # Add embedding vectors to the index
    
    faiss.write_index(index, os.path.join(index_dir, 'chunks.faiss'))    # Save index to file
    
    with open(os.path.join(index_dir, 'chunks_meta.json'), 'w', encoding='utf-8') as f:    # Save metadata (for future retrieval)
        json.dump(chunks_meta, f, ensure_ascii=False)

    print(f"FAISS index successfully built and saved in {index_dir} !")
