import os
import sys
from flask import Flask, render_template, request, jsonify

# 👇 Make sure this matches your actual project root
sys.path.append("/home/cdsw/genai_pdf_chatbot")

from src.query_engine import get_answer_from_query

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    response = get_answer_from_query(user_input)
    return jsonify({"response": response})


# Start the app in CML environment
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ["CDSW_APP_PORT"]))