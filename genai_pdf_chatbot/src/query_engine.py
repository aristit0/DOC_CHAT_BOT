from sentence_transformers import SentenceTransformer
import numpy as np
from src.vector_store import load_index
from src.genai import generate_response

INDEX_DIR = "genai_pdf_chatbot/embeddings/faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("🔍 Loading model and index...")
model = SentenceTransformer(MODEL_NAME)
model.to("cuda")
index, texts, metadata = load_index(INDEX_DIR)

def get_answer_from_query(query, top_k=3):
    print(f"🔎 Searching for top {top_k} relevant chunks...")
    query_vector = model.encode([query], device="cuda")
    distances, indices = index.search(np.array(query_vector), top_k)

    context = "\n\n".join([texts[i] for i in indices[0]])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    return generate_response(prompt)