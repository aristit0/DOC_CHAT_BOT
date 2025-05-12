import os
import sys
import fitz  # PyMuPDF
import re
import numpy as np
from sentence_transformers import SentenceTransformer

# Make sure src/ is on the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vector_store import save_index

DOCUMENTS_DIR = "data/documents"
INDEX_DIR = "embeddings/faiss_index"
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
    model = SentenceTransformer(MODEL_NAME)
    model.to("cuda")
    texts, metadata = [], []

    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.endswith(".pdf"):
            print(f"Processing {filename}...")
            full_path = os.path.join(DOCUMENTS_DIR, filename)
            text = clean_text(extract_text_from_pdf(full_path))
            chunks = chunk_text(text)
            for chunk in chunks:
                texts.append(chunk)
                metadata.append({"source": filename})

    embeddings = model.encode(texts, show_progress_bar=True, device="cuda")
    save_index(embeddings, texts, metadata, INDEX_DIR)

if __name__ == "__main__":
    process_documents()