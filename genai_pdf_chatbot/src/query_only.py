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

# --- Loaders ---
def load_index():
    index = faiss.read_index(os.path.join(INDEX_DIR, "docs.index"))
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "rb") as f:
        data = pickle.load(f)
    return index, data["texts"], data["meta"]

# --- LLM Pipeline ---
generator = pipeline("text-generation", model=LLM_NAME, device=0 if DEVICE == "cuda" else -1)

def generate_response(prompt, max_tokens=256):
    output = generator(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
    return output[0]["generated_text"].split("Answer:")[-1].strip()

# --- Query Logic ---
def get_answer_from_query(query, top_k=3):
    model = SentenceTransformer(MODEL_NAME)
    model.to(DEVICE)

    index, texts, metadata = load_index()
    query_vector = model.encode([query], device=DEVICE)
    distances, indices = index.search(np.array(query_vector), top_k)

    context = "\n\n".join([texts[i] for i in indices[0]])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    return generate_response(prompt)

# --- Entry Point ---
# Manual query input for CML notebook
question = "What is the purpose of this document?"  # change this as needed
answer = get_answer_from_query(question)
print("\n🤖 Answer:\n" + answer)