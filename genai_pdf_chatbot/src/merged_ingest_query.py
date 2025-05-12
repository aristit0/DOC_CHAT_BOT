import os
import sys
import fitz  # PyMuPDF
import re
import time
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# === CONFIGURATION ===
DOCUMENTS_DIR = "genai_pdf_chatbot/data/documents"
INDEX_DIR = "genai_pdf_chatbot/embeddings/faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME = "tiiuae/falcon-7b-instruct"
DEVICE = "cuda"  # use "cpu" if needed

# === STEP 1: PDF LOADING AND CHUNKING ===

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text, max_tokens=256):
    sentences = text.split(". ")
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < max_tokens:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# === STEP 2: FAISS VECTOR STORE ===

def save_index(embeddings, texts, metadata):
    os.makedirs(INDEX_DIR, exist_ok=True)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    faiss.write_index(index, os.path.join(INDEX_DIR, "docs.index"))
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump({"texts": texts, "meta": metadata}, f)
    print("💾 FAISS index and metadata saved.")

def load_index():
    index = faiss.read_index(os.path.join(INDEX_DIR, "docs.index"))
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "rb") as f:
        data = pickle.load(f)
    return index, data["texts"], data["meta"]

# === STEP 3: TEXT GENERATION (GENAI) ===

generator = pipeline("text-generation", model=LLM_NAME, device=0 if DEVICE == "cuda" else -1)

def generate_response(prompt, max_tokens=256):
    output = generator(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
    return output[0]["generated_text"].split("Answer:")[-1].strip()

# === STEP 4: INGEST FUNCTION ===

def process_documents():
    print("⚙️ Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    model.to(DEVICE)

    texts, metadata = [], []

    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.endswith(".pdf"):
            print(f"\n📄 Processing {filename}...")
            full_path = os.path.join(DOCUMENTS_DIR, filename)
            text = clean_text(extract_text_from_pdf(full_path))
            print(f"📜 Extracted {len(text)} characters.")
            chunks = chunk_text(text)
            print(f"✂️  Chunked into {len(chunks)} segments.")
            texts.extend(chunks)
            metadata.extend([{"source": filename}] * len(chunks))

    if not texts:
        print("⚠️ No text to embed.")
        return

    print(f"\n🧠 Embedding {len(texts)} chunks...")
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        device=DEVICE
    )
    print(f"✅ Embedding done in {round(time.time() - start, 2)}s.")
    save_index(embeddings, texts, metadata)

# === STEP 5: QUERY FUNCTION ===

def get_answer_from_query(query, top_k=3):
    print("🔎 Searching index...")
    model = SentenceTransformer(MODEL_NAME)
    model.to(DEVICE)

    index, texts, metadata = load_index()
    query_vector = model.encode([query], device=DEVICE)
    distances, indices = index.search(np.array(query_vector), top_k)

    context = "\n\n".join([texts[i] for i in indices[0]])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    return generate_response(prompt)

# === MAIN SCRIPT LOGIC ===

if __name__ == "__main__":
    print("=== 📘 GenAI PDF Ingest & Query Tool ===")
    mode = input("Enter mode (ingest/query): ").strip().lower()

    if mode == "ingest":
        process_documents()

    elif mode == "query":
        question = input("Ask your question: ")
        answer = get_answer_from_query(question)
        print("\n🤖 Answer:\n" + answer)

    else:
        print("❌ Unknown mode. Use 'ingest' or 'query'.")