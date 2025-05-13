# 🧠 GenAI PDF Chatbot

A simple web-based chatbot that lets users upload PDFs, ask questions about their content, and manually add knowledge—all in a clean, intuitive interface.

---

## 🚀 Features

- 📄 Upload and embed PDF documents
- 💬 Ask natural language questions about uploaded files
- 🧠 Manually input new knowledge via textarea
- 📎 Clean and responsive UI with real-time feedback
- 🧹 Option to clear chat history

---

## 📂 App Flow

### 1. Upload PDF

- Select a `.pdf` file in the **Upload a PDF** section
- Click **📤 Upload & Embed**
- File is sent to the `/upload` endpoint
- Shows `🔄 Uploading and embedding...` while uploading
- On success:
  - Success message shown for 2 seconds
  - File input is cleared

### 2. Chat with Your Document

- Type a question into the chat input
- Click **➡️** to submit
- The message is displayed in the chat window
- Sent to the `/chat` endpoint as JSON
- Bot response appears below
- Click **🧹 Clear** to reset the conversation (calls `/clear_chat`)

### 3. Add New Knowledge

- Use the **Add New Knowledge** section
- Enter any custom knowledge in the textarea
- Click **🧠 Submit** to send to `/add_knowledge`
- Textarea is cleared on success
- Status message disappears after 2 seconds

---

## 🖼️ UI Overview

```text
📄 GenAI PDF Chatbot
├── Upload a PDF
│   └── File input + Submit button + Status message
├── Chat with Your Document
│   └── Chat window + User input + Submit + Clear
└── Add New Knowledge
    └── Textarea + Submit button + Status message