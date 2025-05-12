import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# --- Config ---
INDEX_DIR = "genai_pdf_chatbot/embeddings/faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME = "tiiuae/falcon-7b-instruct"
DEVICE = "cuda"

# --- Load embedding model ONCE ---
print("🔁 Loading embedding model...")
embedding_model = SentenceTransformer(MODEL_NAME)
embedding_model.to(DEVICE)

# --- Load FAISS index + metadata ONCE ---
def load_index_once():
    print("📦 Loading FAISS index and metadata...")
    index = faiss.read_index(os.path.join(INDEX_DIR, "docs.index"))
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "rb") as f:
        data = pickle.load(f)
    return index, data["texts"], data["meta"]

index, texts, metadata = load_index_once()

# --- Load LLM generator ONCE ---
print("⚙️ Loading text generation model...")
generator = pipeline("text-generation", model=LLM_NAME, device=0 if DEVICE == "cuda" else -1)

def generate_response(prompt, max_tokens=256):
    output = generator(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
    return output[0]["generated_text"].split("Answer:")[-1].strip()

# --- Query Logic ---
def get_answer_from_query(query, top_k=3):
    query_vector = embedding_model.encode([query], device=DEVICE)
    distances, indices = index.search(np.array(query_vector), top_k)

    context = "\n\n".join([texts[i] for i in indices[0]])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    return generate_response(prompt)

# --- Notebook usage (no input()) ---
if __name__ == "__main__":
    # You can call this manually like:
    # get_answer_from_query("What is the purpose of this document?")
    print("✅ Query interface loaded. Use `get_answer_from_query('your question')` to interact.")