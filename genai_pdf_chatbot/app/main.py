import os
import sys
from flask import Flask, render_template, request, jsonify

# Add your project root so you can import from src/
sys.path.append("/home/cdsw/genai_pdf_chatbot")

from src.query_engine import get_answer_from_query

# ✅ Tell Flask where to find templates, since main.py is in /app/
app = Flask(
    __name__,
    template_folder="/home/cdsw/genai_pdf_chatbot/templates",
    static_folder="/home/cdsw/genai_pdf_chatbot/static"
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    response = get_answer_from_query(user_input)
    return jsonify({"response": response})

# Run using CML App port
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ["CDSW_APP_PORT"]))