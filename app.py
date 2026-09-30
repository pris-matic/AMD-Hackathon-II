import json
import os
import requests
from dotenv import load_dotenv
from llama_cpp import Llama
from flask import Flask, request, jsonify

load_dotenv()  # loads .env when running locally; no effect if .env doesn't exist

app = Flask(__name__)

MODEL_PATH = "/app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")
FIREWORKS_BASE_URL = os.environ.get("FIREWORKS_BASE_URL")
ALLOWED_MODELS = os.environ.get("ALLOWED_MODELS", "").split(",") if os.environ.get("ALLOWED_MODELS") else []

# Categories we trust the small local model with.
# Everything else routes to Fireworks.
LOCAL_KEYWORDS = ["sentiment", "classify", "summar"]

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=2,
    verbose=False,
)

def should_use_local(prompt):
    p = prompt.lower()
    return any(keyword in p for keyword in LOCAL_KEYWORDS)

def call_local(prompt):
    output = llm(
        f"Answer the question in 1-3 sentences. Be concise and factual.\nQuestion: {prompt}\nAnswer:",
        max_tokens=120,
        temperature=0.2,
        repeat_penalty=1.3,
        stop=["\n\n", "Question:"],
    )
    return output["choices"][0]["text"].strip()

def call_fireworks(prompt):
    model = ALLOWED_MODELS[0]
    response = requests.post(
        f"{FIREWORKS_BASE_URL.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {FIREWORKS_API_KEY}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150,
            "temperature": 0.2,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()

@app.route("/process", methods=["POST"])
def process():
    try:
        data = request.json
        if not data or "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' field"}), 400
        
        prompt = data.get("prompt")
        
        if should_use_local(prompt):
            answer = call_local(prompt)
        else:
            if FIREWORKS_API_KEY and FIREWORKS_BASE_URL and ALLOWED_MODELS:
                answer = call_fireworks(prompt)
            else:
                answer = call_local(prompt)
        
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
