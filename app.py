from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import json
import os
import uuid
from urllib.parse import quote

from rag_pipeline import setup_rag_pipeline, process_query, session_memories

app = Flask(__name__)
app.secret_key = 'super_secret_key'

# ==== User Auth Logic ====

USER_FILE = "users.json"
if os.path.exists(USER_FILE):
    with open(USER_FILE) as f:
        USERS = json.load(f)
else:
    USERS = {}

def save_users():
    with open(USER_FILE, "w") as f:
        json.dump(USERS, f)

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in USERS:
            return "Username already exists", 400

        USERS[username] = {"password_hash": generate_password_hash(password)}
        save_users()
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = USERS.get(username)

        if user and check_password_hash(user["password_hash"], password):
            session["username"] = username
            return redirect(url_for("index"))

        return "Invalid credentials", 401

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ==== Main Chatbot Page ====

@app.route("/")
@login_required
def index():
    return render_template("index.html", username=session["username"])

# ==== PDF Routes ====

@app.route("/pdf-list", methods=["GET"])
def pdf_list():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = os.path.join(base_dir, "data", "pdf_files")

    if not os.path.exists(pdf_dir):
        return jsonify({"error": "PDF folder not found"}), 500

    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    return jsonify({"pdfs": pdf_files})

@app.route("/pdf/<path:filename>")
def serve_pdf(filename):
    pdf_dir = os.path.join("data", "pdf_files")
    return send_from_directory(pdf_dir, filename)

# ==== Ask Query ====

@app.route("/ask", methods=["POST"])
@login_required
def ask():
    data = request.get_json()
    query = data.get("query", "")
    pdf_name = data.get("pdf_name", None)
    if not query:
        return jsonify({"error": "Empty query"}), 400

    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

    answer, chunks = process_query(query, session['session_id'], return_chunks=True, pdf_filter=pdf_name)
    return jsonify({"answer": answer, "chunks": chunks})

# ==== Reset Chat ====

@app.route("/reset", methods=["POST"])
@login_required
def reset_conversation():
    if 'session_id' in session:
        session_id = session['session_id']
        if session_id in session_memories:
            del session_memories[session_id]
    return jsonify({"status": "conversation reset"})

# ==== Init RAG on startup ====

setup_rag_pipeline()

if __name__ == "__main__":
    app.run(debug=True)
