import os
import sys
import requests
from flask import Flask, render_template, request, jsonify

# Add your project root so you can import from src/
sys.path.append("/home/cdsw/genai_pdf_chatbot")

from src.query_engine import get_answer_from_query

# === Configuration ===
UPLOAD_FOLDER = "/home/cdsw/genai_pdf_chatbot/data/documents"
CDSW_DOMAIN = os.getenv("CDSW_DOMAIN")       # e.g., ml-yourcluster.cloudera.site
API_KEY = os.getenv("CDSW_API_KEY")          # from your user API keys
INGEST_JOB_ID = "m4in-jwsz-ltn3-vq2u"        # your job ID from Cloudera UI

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

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(save_path)
    print(f"✅ Saved file to {save_path}")

    # Trigger the CML job
    headers = {"Authorization": f"Bearer {API_KEY}"}
    trigger_url = f"https://{CDSW_DOMAIN}/api/v2/jobs/{INGEST_JOB_ID}/runs"
    res = requests.post(trigger_url, headers=headers)

    if res.status_code in [200, 201]:
        try:
            run_id = res.json().get("id")
        except Exception:
            run_id = None

        return jsonify({
            "message": "File uploaded. Ingest job started ✅",
            "run_id": run_id
        }), 200
    else:
        try:
            error_details = res.json()
        except Exception:
            error_details = res.text or "No response body"

        return jsonify({
            "error": "Upload succeeded but job trigger failed",
            "status": res.status_code,
            "details": error_details
        }), 500


@app.route("/job_status/<run_id>")
def job_status(run_id):
    status_url = f"https://{CDSW_DOMAIN}/api/v2/runs/{run_id}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    res = requests.get(status_url, headers=headers)

    if res.status_code == 200:
        data = res.json()
        return jsonify({
            "status": data.get("status"),
            "start_time": data.get("start_time"),
            "end_time": data.get("end_time")
        })
    else:
        return jsonify({
            "error": f"Failed to fetch status for run {run_id}",
            "details": res.text
        }), 500


# === Run in CML ===
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ["CDSW_APP_PORT"]))