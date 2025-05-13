import os
import sys
import requests
from flask import Flask, render_template, request, jsonify

# Add your project root so you can import from src/
sys.path.append("/home/cdsw/genai_pdf_chatbot")

from src.query_engine import get_answer_from_query

# === Configuration ===
UPLOAD_FOLDER = "/home/cdsw/genai_pdf_chatbot/data/documents"
CDSW_DOMAIN = os.getenv("CDSW_DOMAIN")  # e.g., ml-abc123.cloudera.site
API_KEY = os.getenv("CDSW_API_KEY")     # from CML environment variables
INGEST_JOB_ID = "m4in-jwsz-ltn3-vq2u"   # your specific ingest job ID

# === Flask App Setup ===
app = Flask(
    __name__,
    template_folder="/home/cdsw/genai_pdf_chatbot/templates",
    static_folder="/home/cdsw/genai_pdf_chatbot/static"
)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    response = get_answer_from_query(user_input)
    return jsonify({"response": response})


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save the uploaded file
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(save_path)
    print(f"✅ Saved file to {save_path}")

    # Trigger ingest job via CML API v2
    headers = {"Authorization": f"Bearer {API_KEY}"}
    url = f"https://{CDSW_DOMAIN}/api/v2/jobs/{INGEST_JOB_ID}/runs"

    res = requests.post(url, headers=headers)

    if res.status_code in [200, 201]:
        return jsonify({"message": f"File uploaded and ingest job triggered ✅"}), 200
    else:
        return jsonify({
            "error": "Upload succeeded but job trigger failed",
            "status": res.status_code,
            "details": res.json()
        }), 500


# === Run App in CML ===
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ["CDSW_APP_PORT"]))