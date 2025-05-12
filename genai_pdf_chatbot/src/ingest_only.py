import os
import sys
import fitz  # PyMuPDF
import re
import time
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- Config ---
DOCUMENTS_DIR = "genai_pdf_chatbot/data/documents"
INDEX_DIR = "genai_pdf_chatbot/embeddings/faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda"

# --- Functions ---
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

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

def save_index(embeddings, texts, metadata):
    os.makedirs(INDEX_DIR, exist_ok=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    faiss.write_index(index, os.path.join(INDEX_DIR, "docs.index"))
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump({"texts": texts, "meta": metadata}, f)
    print("💾 Saved FAISS index and metadata.")

def process_documents():
    model = SentenceTransformer(MODEL_NAME)
    model.to(DEVICE)

    texts, metadata = [], []

    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.endswith(".pdf"):
            print(f"\n📄 Processing {filename}...")
            text = clean_text(extract_text_from_pdf(os.path.join(DOCUMENTS_DIR, filename)))
            chunks = chunk_text(text)
            print(f"🧠 {len(chunks)} chunks.")
            texts.extend(chunks)
            metadata.extend([{"source": filename}] * len(chunks))

    if not texts:
        print("⚠️ No text to embed.")
        return

    print(f"🚀 Embedding {len(texts)} chunks...")
    start = time.time()
    embeddings = model.encode(texts, batch_size=16, show_progress_bar=True, device=DEVICE)
    print(f"✅ Done in {round(time.time() - start, 2)}s.")
    save_index(embeddings, texts, metadata)

# --- Entry point ---
if __name__ == "__main__":
    process_documents()