
import hashlib
import time
import os
import openai
from flask import Flask, request, jsonify

app = Flask(__name__)
CACHE = {}
CACHE_TTL = 60 * 60  # 1 hour
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not set. /summarize will return an error until it's provided.")
else:
    openai.api_key = OPENAI_API_KEY
def cache_get(key):
    entry = CACHE.get(key)
    if not entry:
        return None
    ts, value = entry
    if time.time() - ts > CACHE_TTL:
        del CACHE[key]
        return None
    return value

def cache_set(key, value):
    CACHE[key] = (time.time(), value)
# Simple GET to prove the server is alive
@app.route("/hello", methods=["GET"])
def hello():
    return jsonify({"message": "Hello from your API", "status": "ok"}), 200

# POST endpoint that echoes back the received JSON
@app.route("/echo", methods=["POST"])
def echo():
    # 1. JSON must exist
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    
    data = request.get_json()

    # 2. Must contain the key "name"
    if "name" not in data:
        return jsonify({"error": "'name' field is required"}), 400
    
    # 3. Must be a string
    if not isinstance(data["name"], str):
        return jsonify({"error": "'name' must be a string"}), 400

    # If valid → success response
    return jsonify({
        "received": data,
        "message": "Valid input received",
    }), 200

@app.route("/summarize", methods=["POST"])
def summarize():
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400

    data = request.get_json()

    if "text" not in data:
        return jsonify({"error": "'text' field is required"}), 400

    if not isinstance(data["text"], str):
        return jsonify({"error": "'text' must be a string"}), 400

    if not OPENAI_API_KEY:
        return jsonify({"error": "OpenAI API key not configured on the server"}), 500

    text = data["text"].strip()
    if text == "":
        return jsonify({"error": "'text' must be a non-empty string"}), 400

    # Build a deterministic-ish prompt for summarization
    system_prompt = (
        "You are a concise summarizer. Produce a short, factual, bullet-point summary "
        "of the user's text. Do not invent facts. If the text is instructions or list-like, "
        "preserve the most important action items. Keep the summary to 3-6 bullets when possible."
    )

    user_prompt = f"Summarize the following text:\n\n{text}"

    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=256,
            temperature=0.2,
            n=1,
            timeout=30
        )
        summary_text = resp["choices"][0]["message"]["content"].strip()

        lines = [line.strip() for line in summary_text.splitlines() if line.strip()]
        if len(lines) <= 1:
            import re
            sentences = re.split(r'(?<=[.!?])\s+', summary_text)
            lines = [s.strip() for s in sentences if s.strip()][:6]

        return jsonify({
            "summary_raw": summary_text,
            "summary_bullets": lines
        }), 200

    except Exception as e:
        from openai.error import OpenAIError
        if isinstance(e, OpenAIError):
            return jsonify({"error": "OpenAI API error", "detail": str(e)}), 502
        return jsonify({"error": "Server error", "detail": str(e)}), 500

@app.route("/keywords", methods=["POST"])
def keywords():
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400

    data = request.get_json()
    if "text" not in data:
        return jsonify({"error": "'text' field is required"}), 400
    if not isinstance(data["text"], str):
        return jsonify({"error": "'text' must be a string"}), 400

    text = data["text"].strip()
    if text == "":
        return jsonify({"error": "'text' must be a non-empty string"}), 400

    # input length guard
    MAX_CHARS = 4000
    if len(text) > MAX_CHARS:
        return jsonify({"error": "input_too_long", "detail": f"Text too long ({len(text)} chars). Max {MAX_CHARS}."}), 400

    if not OPENAI_API_KEY:
        return jsonify({"error": "OpenAI API key not configured on the server"}), 500

    # caching
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cached = cache_get(key)
    if cached:
        return jsonify(cached), 200

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = (
        "You are a strict JSON generator. Given input text, return ONLY a JSON array "
        "of up to 5 keywords (strings). Output must be valid JSON with no surrounding text, "
        "no explanation, and no extra characters. Examples: [\"ai\",\"machine learning\"]"
    )
    user_prompt = f"Extract up to 5 keywords from the following text. Text:\n\n{text}"

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=150,
            temperature=0.0,
            n=1,
            timeout=30
        )

        try:
            raw = resp.choices[0].message.content.strip()
        except Exception:
            raw = resp["choices"][0]["message"]["content"].strip()

        import json, re
        m = re.search(r'(\[.*?\])', raw, flags=re.S)
        if m:
            arr_text = m.group(1)
            try:
                keywords_list = json.loads(arr_text)
            except Exception:
                cleaned = re.split(r'[\n,]+', arr_text.strip('[] '))
                keywords_list = [k.strip().strip('"').strip("'") for k in cleaned if k.strip()]
        else:
            parts = re.split(r'[\n,]+', raw)
            keywords_list = [p.strip().strip('-• ').strip() for p in parts if p.strip()][:5]

        keywords_list = [str(k) for k in keywords_list][:5]

        result = {"keywords_raw": raw, "keywords": keywords_list}
        cache_set(key, result)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": "Server error", "detail": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)