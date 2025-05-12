import os
import sys
import fitz  # PyMuPDF
import re
import time
import numpy as np
from sentence_transformers import SentenceTransformer

# Ensure `src/` path is included
sys.path.append("/home/cdsw/genai_pdf_chatbot/")

from src.vector_store import save_index

DOCUMENTS_DIR = "genai_pdf_chatbot/data/documents"
INDEX_DIR = "genai_pdf_chatbot/embeddings/faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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

def process_documents():
    print("⚙️ Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    model.to("cuda")

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

    print(f"\n🧠 Total chunks to embed: {len(texts)}")
    if not texts:
        print("⚠️ No text found to embed. Exiting.")
        return

    print("🚀 Starting embedding on GPU...")
    start = time.time()
    try:
        embeddings = model.encode(
            texts,
            batch_size=16,
            show_progress_bar=True,
            device="cuda"
        )
    except Exception as e:
        print(f"🔥 GPU embedding failed: {e}")
        print("🛑 Falling back to CPU...")
        embeddings = model.encode(
            texts,
            batch_size=8,
            show_progress_bar=True,
            device="cpu"
        )

    print(f"✅ Embedding complete in {round(time.time() - start, 2)}s.")
    save_index(embeddings, texts, metadata, INDEX_DIR)
    print("💾 Index saved. Done!")

if __name__ == "__main__":
    process_documents()