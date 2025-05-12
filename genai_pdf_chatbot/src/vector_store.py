import os
import faiss
import pickle
import numpy as np

def save_index(embeddings, texts, metadata, index_dir):
    os.makedirs(index_dir, exist_ok=True)
    dim = embeddings.shape[1]

    # Use GPU FAISS
    res = faiss.StandardGpuResources()
    cpu_index = faiss.IndexFlatL2(dim)
    index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    index.add(np.array(embeddings))
    faiss.write_index(faiss.index_gpu_to_cpu(index), os.path.join(index_dir, "docs.index"))

    with open(os.path.join(index_dir, "metadata.pkl"), "wb") as f:
        pickle.dump({"texts": texts, "meta": metadata}, f)

def load_index(index_dir):
    cpu_index = faiss.read_index(os.path.join(index_dir, "docs.index"))
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    with open(os.path.join(index_dir, "metadata.pkl"), "rb") as f:
        data = pickle.load(f)
    return index, data["texts"], data["meta"]