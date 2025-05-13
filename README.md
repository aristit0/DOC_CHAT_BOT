# DOC_CHAT_BOT
# 🧠 GenAI PDF Chatbot

A simple web-based chatbot that allows users to upload PDFs, ask questions about their contents, and manually add knowledge. Built with Flask, JavaScript, and a friendly UI.

---

## 🚀 Features

- 📄 Upload and embed PDF documents
- 💬 Ask natural language questions about uploaded files
- 🧠 Manually input new knowledge via textarea
- 📎 Clean and intuitive UI with status messages
- 🧹 Option to clear chat history

---

## 📂 App Flow

### 1. Upload PDF
- Go to the **Upload a PDF** section.
- Select a `.pdf` file and click **📤 Upload & Embed**.
- The file is sent to the `/upload` endpoint via `POST`.
- A loading message (`🔄 Uploading and embedding...`) appears.
- On success, the input resets and shows a success message (auto-hides after 2 seconds).

### 2. Chat with Your Document
- Enter a question in the chat input field and click **➡️**.
- The message appears in the chat window and is sent to `/chat` via `POST`.
- The bot returns a response based on the document contents.
- Click **🧹 Clear** to reset the conversation (calls `/clear_chat`).

### 3. Add New Knowledge
- Scroll to the **Add New Knowledge** section.
- Type a fact, insight, or manual information into the textarea.
- Click **🧠 Submit** to send it to `/add_knowledge` as JSON.
- A confirmation message will show and the input will be cleared.
- Status auto-clears after 2 seconds.

---

## 📁 Project Structure (Frontend)
/templates/
└── index.html          # Main HTML file with upload, chat, and manual input sections
/static/
├── style.css           # Clean and responsive styles
└── chatbot.png         # Bot avatar image (optional)
/app.py (or main Flask backend)
---


## 🛠️ Requirements (Backend)

- Python 3.9+
- Flask
- Flask-CORS
- PDF parser (like PyMuPDF or PDFMiner)
- Embedding + Vector search (e.g. FAISS)
- OpenAI or local LLM for response generation

---