import os
import sys
import time
import re
import pickle
import requests
import fitz  # PyMuPDF
import numpy as np
import faiss
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# === Configuration ===
UPLOAD_FOLDER = "/home/cdsw/genai_pdf_chatbot/data/documents"
INDEX_DIR = "/home/cdsw/genai_pdf_chatbot/embeddings/faiss_index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME = "tiiuae/falcon-7b-instruct"
DEVICE = "cuda"

# === Flask App Setup ===
app = Flask(
    __name__,
    template_folder="/home/cdsw/genai_pdf_chatbot/templates",
    static_folder="/home/cdsw/genai_pdf_chatbot/static"
)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# === Chat memory ===
chat_history = []

# --- Load Models Once ---
print("🔁 Loading embedding model...")
embedding_model = SentenceTransformer(MODEL_NAME)
embedding_model.to(DEVICE)

print("📦 Loading FAISS index and metadata...")
def load_index():
    index = faiss.read_index(os.path.join(INDEX_DIR, "docs.index"))
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "rb") as f:
        data = pickle.load(f)
    return index, data["texts"], data["meta"]

try:
    index, texts, metadata = load_index()
except Exception as e:
    print(f"⚠️ FAISS index load failed: {e}")
    index, texts, metadata = None, [], []

print("⚙️ Loading text generation model...")
generator = pipeline("text-generation", model=LLM_NAME, device=0 if DEVICE == "cuda" else -1)

# === Web Routes ===
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    response = get_answer_from_query(user_input)
    chat_history.append({"user": user_input, "bot": response})
    return jsonify({"response": response})


@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    chat_history.clear()
    return jsonify({"message": "Chat cleared."})


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(save_path)
    print(f"✅ Saved file to {save_path}")

    status = process_documents()
    if status == "done":
        global index, texts, metadata
        index, texts, metadata = load_index()
        return jsonify({"message": "File uploaded and embedded ✅"}), 200
    else:
        return jsonify({"error": "Failed during embedding."}), 500


# === Processing Functions ===
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
    try:
        model = embedding_model
        texts, metadata = [], []

        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.endswith(".pdf"):
                print(f"\n📄 Processing {filename}...")
                text = clean_text(extract_text_from_pdf(os.path.join(UPLOAD_FOLDER, filename)))
                chunks = chunk_text(text)
                print(f"🧠 {len(chunks)} chunks.")
                texts.extend(chunks)
                metadata.extend([{"source": filename}] * len(chunks))

        if not texts:
            print("⚠️ No text to embed.")
            return "empty"

        print(f"🚀 Embedding {len(texts)} chunks...")
        start = time.time()
        embeddings = model.encode(texts, batch_size=16, show_progress_bar=True, device=DEVICE)
        print(f"✅ Done in {round(time.time() - start, 2)}s.")
        save_index(embeddings, texts, metadata)
        return "done"
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return "error"


def generate_response(prompt, max_tokens=256):
    output = generator(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
    return output[0]["generated_text"].split("Answer:")[-1].strip()


def get_answer_from_query(query, top_k=3):
    if index is None:
        return "❌ No embedded documents found. Please upload a PDF first."

    query_vector = embedding_model.encode([query], device=DEVICE)
    distances, indices = index.search(np.array(query_vector), top_k)

    threshold = 1.0
    retrieved = [(i, d) for i, d in zip(indices[0], distances[0]) if d < threshold]
    if not retrieved:
        return "❌ Sorry, I couldn't find anything relevant in your documents."

    context = "\n\n".join([texts[i] for i, _ in retrieved])
    prompt = f"""
Use ONLY the information in the context below to answer the question.
If the answer is not in the context, reply: "❌ Sorry, I couldn't find that in your documents."

Context:
{context}

Question: {query}
Answer:"""
    return generate_response(prompt)


# === Run in CML ===
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ["CDSW_APP_PORT"]))