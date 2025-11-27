
import hashlib
import time
import os
import json
import math
import sqlite3
import re
from typing import Dict, List

import openai
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, Response
# Simple in-memory vector store
RAG_STORE = []


app = Flask(__name__)
# ensure Flask's JSON responses do not escape non-ascii characters
app.config['JSON_AS_ASCII'] = False
# Simple in-memory cache for responses
CACHE = {}
CACHE_TTL = 60 * 60  # 1 hour

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not set. /summarize will return an error until it's provided.")
else:
    openai.api_key = OPENAI_API_KEY


def cache_get(key):
    """Fetch a cached value by key, respecting TTL."""
    entry = CACHE.get(key)
    if not entry:
        return None

    ts, value = entry
    if time.time() - ts > CACHE_TTL:
        # Expired, remove and treat as a miss
        del CACHE[key]
        return None

    return value


def cache_set(key, value):
    """Store a value in the in-memory cache with the current timestamp."""
    CACHE[key] = (time.time(), value)


# Simple GET to prove the server is alive
@app.route("/hello", methods=["GET"])
def hello():
    """Health check endpoint."""
    return jsonify({"message": "Hello from your API", "status": "ok"}), 200

# POST endpoint that echoes back the received JSON
@app.route("/echo", methods=["POST"])
def echo():
    """Echo back validated JSON payload."""

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
    """Summarize the provided text using OpenAI."""

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
    """Extract up to 5 important keywords from the text."""

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

@app.route("/sentiment", methods=["POST"])
def sentiment():
    """Classify sentiment (positive / neutral / negative) for the given text."""

    # Validate JSON input
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

    # Input length guard (prevent huge token bills)
    MAX_CHARS_SENTIMENT = 4000
    if len(text) > MAX_CHARS_SENTIMENT:
        return jsonify({"error": "input_too_long", "detail": f"Text too long ({len(text)} chars). Max {MAX_CHARS_SENTIMENT}."}), 400

    if not OPENAI_API_KEY:
        return jsonify({"error": "OpenAI API key not configured on the server"}), 500

    # Use the same in-memory cache pattern (sha256 text)
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cached = cache_get(key)
    if cached:
        return jsonify(cached), 200

    # Create OpenAI client (new-style)
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Prompt: ask model to return only JSON with 'label' and 'score'
    system_prompt = (
        "You are a classifier. Given the user's text, return ONLY a JSON object "
        "with two keys: 'label' and 'score'. 'label' must be one of: positive, neutral, negative. "
        "'score' must be a number between 0.00 and 1.00 representing confidence. "
        "Output must be valid JSON and nothing else. Example: {\"label\":\"positive\",\"score\":0.92}"
    )
    user_prompt = f"Classify the sentiment of this text and return the JSON object:\n\n{text}"

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=64,
            temperature=0.0,
            n=1,
            timeout=30
        )

        # get raw model output text
        try:
            raw = resp.choices[0].message.content.strip()
        except Exception:
            raw = resp["choices"][0]["message"]["content"].strip()

        # Try to parse JSON from model output
        import json, re
        m = re.search(r'(\{.*\})', raw, flags=re.S)
        if m:
            obj_text = m.group(1)
            try:
                parsed = json.loads(obj_text)
            except Exception:
                # fallback parsing: attempt to extract label and score naively
                parsed = {}
                lab = re.search(r'"?label"?\s*[:=]\s*"?(positive|neutral|negative)"?', raw, flags=re.I)
                sc = re.search(r'"?score"?\s*[:=]\s*([0-9]*\.?[0-9]+)', raw)
                if lab:
                    parsed["label"] = lab.group(1).lower()
                if sc:
                    parsed["score"] = float(sc.group(1))
        else:
            # fallback: attempt naive extraction
            parsed = {}
            lab = re.search(r'(positive|neutral|negative)', raw, flags=re.I)
            sc = re.search(r'([0-9]*\.?[0-9]+)', raw)
            if lab:
                parsed["label"] = lab.group(1).lower()
            if sc:
                parsed["score"] = float(sc.group(1))

        # Validate parsed output and normalize
        label = parsed.get("label")
        score = parsed.get("score")

        if label not in ("positive", "neutral", "negative"):
            # If label invalid or missing, infer a neutral fallback
            label = "neutral"

        # Ensure score is a float in [0.0, 1.0]
        try:
            score = float(score)
            if score < 0.0 or score > 1.0:
                score = max(0.0, min(1.0, score))
        except Exception:
            # If no numeric score, set default based on label
            score = {"positive": 0.9, "neutral": 0.5, "negative": 0.9}.get(label, 0.5)

        result = {
            "sentiment_raw": raw,
            "label": label,
            "score": round(score, 3)
        }

        # cache & return
        cache_set(key, result)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": "Server error", "detail": str(e)}), 500


@app.route("/entities", methods=["POST"])
def entities():
    """Extract named entities grouped by type."""

    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400

    data = request.get_json()

    if "text" not in data:
        return jsonify({"error": "'text' field is required"}), 400

    if not isinstance(data["text"], str):
        return jsonify({"error": "'text' must be a string"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "'text' must be non-empty"}), 400

    if not OPENAI_API_KEY:
        return jsonify({"error": "OpenAI API key not configured"}), 500

    try:
        system_prompt = (
            "Extract named entities from the text. "
            "Return ONLY valid JSON with this structure:\n"
            "{\n"
            '  "people": [],\n'
            '  "locations": [],\n'
            '  "organizations": [],\n'
            '  "dates": [],\n'
            '  "other": []\n'
            "}\n"
            "Do not include explanations. Do not add extra keys."
        )

        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )

        raw = resp.choices[0].message.content

        import json

        try:
            parsed = json.loads(raw)
        except Exception:
            return jsonify(
                {
                    "error": "Model returned invalid JSON",
                    "raw": raw,
                }
            ), 502

        return jsonify({"entities": parsed, "raw": raw}), 200

    except Exception as e:
        return jsonify({"error": "Server error", "detail": str(e)}), 500


# ========================================
# 5. /rewrite – Improve text clarity/grammar
# ========================================
@app.route("/rewrite", methods=["POST"])
def rewrite():
    """
    Rewrites user text for better clarity, grammar, and flow using OpenAI.
    """
    # Validate JSON input
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    
    data = request.get_json()

    if "text" not in data:
        return jsonify({"error": "'text' field is required"}), 400

    text = data["text"].strip()
    if not isinstance(text, str) or text == "":
        return jsonify({"error": "'text' must be a non-empty string"}), 400

    if not OPENAI_API_KEY:
        return jsonify({"error": "OpenAI API key not configured on the server"}), 500

    # Create OpenAI client
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = (
        "You are a writing assistant. Improve clarity, grammar, flow, and tone "
        "while preserving the original meaning. Do NOT shorten too much or add facts."
    )

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.2
        )

        rewritten = response.choices[0].message.content.strip()

        return jsonify({"rewritten": rewritten}), 200

    except Exception as e:
        return jsonify({"error": "AI processing error", "detail": str(e)}), 500


# ========================================
# 6. /translate – Translate text to another language
# ========================================
@app.route("/translate", methods=["POST"])
def translate():
    """
    Translates user text to a specified target language using OpenAI.
    """
    # Validate JSON input
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400

    data = request.get_json()
    if "text" not in data or "to" not in data:
        return jsonify({"error": "'text' and 'to' fields are required"}), 400

    text = data["text"]
    to_lang = data["to"].strip()
    
    if not isinstance(text, str) or text.strip() == "":
        return jsonify({"error": "'text' must be a non-empty string"}), 400
    if not isinstance(to_lang, str) or to_lang == "":
        return jsonify({"error": "'to' must be a non-empty language string"}), 400

    # Input length guard
    MAX_CHARS_TRANSLATE = 8000
    if len(text) > MAX_CHARS_TRANSLATE:
        return jsonify({
            "error": "input_too_long",
            "detail": f"Text too long ({len(text)} chars). Max {MAX_CHARS_TRANSLATE}."
        }), 400

    if not OPENAI_API_KEY:
        return jsonify({"error": "OpenAI API key not configured on the server"}), 500

    # Create OpenAI client
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = (
        f"You are a precise translator. Translate the user's text to {to_lang}. "
        "Preserve meaning exactly; do not add or remove information. "
        "Output only the translated text, no commentary."
    )

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
            max_tokens=2000,
            timeout=30
        )

        translated = resp.choices[0].message.content.strip()

        # Return with proper UTF-8 encoding for non-ASCII characters
        body = json.dumps(
            {"translation": translated, "to": to_lang},
            ensure_ascii=False
        )

        return Response(body, content_type="application/json; charset=utf-8"), 200

    except Exception as e:
        return jsonify({"error": "AI processing error", "detail": str(e)}), 500


# ========================================
# 7. /classify – Categorize text into predefined categories
# ========================================
@app.route("/classify", methods=["POST"])
def classify():
    """
    Classifies user text into predefined categories using OpenAI.
    """
    # Validate JSON input
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400

    data = request.get_json()

    if "text" not in data:
        return jsonify({"error": "'text' field is required"}), 400

    text = data["text"]
    if not isinstance(text, str) or text.strip() == "":
        return jsonify({"error": "'text' must be a non-empty string"}), 400

    if not OPENAI_API_KEY:
        return jsonify({"error": "OpenAI API key not configured on the server"}), 500

    # Create OpenAI client
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = (
        "You are a text classifier. "
        "You MUST output a single category label from the following list: "
        "Technology, Finance, Health, Education, Entertainment, Politics, Sports, Business, Other. "
        "Respond *only* with the category name."
    )

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            max_tokens=10,
            temperature=0
        )

        category = resp.choices[0].message.content.strip()

        return jsonify({"category": category}), 200

    except Exception as e:
        return jsonify({"error": "Server error", "detail": str(e)}), 500


# ========================================
# 8. /chat – General-purpose chat endpoint
# ========================================
@app.route("/chat", methods=["POST"])
def chat():
    """
    General-purpose chat endpoint that responds to user messages using OpenAI.
    """
    # Validate JSON input
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400

    data = request.get_json()

    if "message" not in data:
        return jsonify({"error": "'message' field is required"}), 400

    if not isinstance(data["message"], str):
        return jsonify({"error": "'message' must be a string"}), 400

    if not OPENAI_API_KEY:
        return jsonify({"error": "OpenAI API key not configured on the server"}), 500

    user_message = data["message"].strip()

    if user_message == "":
        return jsonify({"error": "'message' must be non-empty"}), 400

    try:
        # Create OpenAI client (local to function is fine)
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=200,
            temperature=0.5,
            timeout=30
        )

        # Extract assistant text
        reply = resp.choices[0].message.content.strip()

        return jsonify({"reply": reply}), 200

    except Exception as e:
        return jsonify({"error": "OpenAI API error", "detail": str(e)}), 502


# ========================================
# 9. /summarize-url – Summarize content from a URL
# ========================================
@app.route("/summarize-url", methods=["POST"])
def summarize_url():
    """
    Fetches content from a URL and summarizes it using OpenAI.
    """
    # Validate JSON input
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400

    data = request.get_json()

    if "url" not in data:
        return jsonify({"error": "'url' field is required"}), 400

    url = data["url"]
    if not isinstance(url, str) or url.strip() == "":
        return jsonify({"error": "'url' must be a non-empty string"}), 400

    if not OPENAI_API_KEY:
        return jsonify({"error": "OpenAI API key not configured on the server"}), 500

    # Fetch the webpage
    try:
        page = requests.get(url, timeout=10)
    except Exception as e:
        return jsonify({"error": "Failed to fetch URL", "detail": str(e)}), 400

    if page.status_code != 200:
        return jsonify({"error": f"URL returned status {page.status_code}"}), 400

    # Extract text from HTML
    soup = BeautifulSoup(page.text, "html.parser")
    text = soup.get_text(separator=" ", strip=True)

    if len(text) < 100:
        return jsonify({"error": "Page text too short or not readable"}), 400

    # Create OpenAI client
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = "Summarize this webpage text in 4–6 bullet points. Keep it factual."

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text[:6000]}  # Keep within safe limits
            ],
            temperature=0.3,
            max_tokens=250
        )

        summary = response.choices[0].message.content.strip()

        return jsonify({
            "url": url,
            "summary": summary
        }), 200

    except Exception as e:
        return jsonify({"error": "OpenAI API error", "detail": str(e)}), 502


# ========================================
# 10. /embed – Generate text embeddings
# ========================================
@app.route("/embed", methods=["POST"])
def embed():
    """
    Input JSON:
      {"text": "some text"}
    or
      {"texts": ["text1", "text2", ...]}

    Output JSON:
      {
        "embeddings": [
          {"input": "text1", "embedding": [...float numbers...]},
          ...
        ],
        "model": "<model-name>"
      }
    """
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400

    data = request.get_json()

    # Accept single text or list of texts
    texts = None
    if "text" in data and isinstance(data["text"], str):
        texts = [data["text"]]
    elif "texts" in data and isinstance(data["texts"], list) and all(isinstance(t, str) for t in data["texts"]):
        texts = data["texts"]
    else:
        return jsonify({"error": "Provide 'text' (string) or 'texts' (list of strings)"}), 400

    # Basic validation
    if len(texts) == 0 or any(len(t.strip()) == 0 for t in texts):
        return jsonify({"error": "Texts must be non-empty strings"}), 400

    # model selection: allow env override
    EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    if not OPENAI_API_KEY:
        return jsonify({"error": "OpenAI API key not configured on server"}), 500

    # create client and call embeddings API using modern SDK
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        # resp.data is a list of objects with .embedding
        out = []
        for i, item in enumerate(resp.data):
            out.append({
                "input": texts[i],
                "embedding_length": len(item.embedding),
                "embedding": item.embedding  # list of floats
            })

        return jsonify({
            "model": EMBEDDING_MODEL,
            "embeddings": out,
            "count": len(out)
        }), 200

    except Exception as e:
        return jsonify({"error": "OpenAI API error", "detail": str(e)}), 502

# ========================================
# 11. RAG helper utilities + endpoints
# ========================================

VECTOR_DB = "vector_store.db"
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
MAX_CONTEXT_CHARS = 4000  # safe limit to attach to prompts

# ---------- DB helpers ----------
def ensure_db():
    conn = sqlite3.connect(VECTOR_DB)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS docs (
        id INTEGER PRIMARY KEY,
        text TEXT NOT NULL,
        embedding TEXT NOT NULL,   -- store JSON list
        metadata TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_doc(text: str, embedding: List[float], metadata: Dict = None):
    conn = sqlite3.connect(VECTOR_DB)
    cur = conn.cursor()
    cur.execute("INSERT INTO docs (text, embedding, metadata) VALUES (?, ?, ?)",
                (text, json.dumps(embedding), json.dumps(metadata or {})))
    conn.commit()
    conn.close()

def load_all_embeddings():
    conn = sqlite3.connect(VECTOR_DB)
    cur = conn.cursor()
    cur.execute("SELECT id, text, embedding, metadata FROM docs")
    rows = cur.fetchall()
    conn.close()
    result = []
    for r in rows:
        result.append({
            "id": r[0],
            "text": r[1],
            "embedding": json.loads(r[2]),
            "metadata": json.loads(r[3])
        })
    return result

# ---------- math helpers ----------
def dot(a: List[float], b: List[float]) -> float:
    return sum(x*y for x,y in zip(a,b))

def norm(a: List[float]) -> float:
    return math.sqrt(sum(x*x for x in a))

def cosine_similarity(a: List[float], b: List[float]) -> float:
    na = norm(a); nb = norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot(a,b) / (na*nb)

# ---------- embedding helper ----------
def embed_texts(texts: List[str]) -> List[List[float]]:
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in resp.data]

# ---------- simple chunker ----------
def chunk_text(text: str, chunk_size_words: int = 120, overlap_words: int = 20) -> List[str]:
    # naive word-based chunker for readability (not token-accurate)
    words = re.split(r"\s+", text.strip())
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size_words]
        chunks.append(" ".join(chunk))
        i += (chunk_size_words - overlap_words)
    return chunks

# ensure DB exists on import
ensure_db()

# ---------- indexing endpoint ----------
@app.route("/rag-index", methods=["POST"])
def rag_index():
    """
    POST JSON:
      {"id": "<optional id or slug>", "text": "<document text>", "metadata": {...} }
    Splits document into chunks, embeds each chunk, stores in SQLite.
    """
    if not request.is_json:
        return jsonify({"error":"JSON body required"}), 400
    payload = request.get_json()
    if "text" not in payload:
        return jsonify({"error":"'text' field required"}), 400
    text = payload["text"]
    metadata = payload.get("metadata", {})
    # split into chunks
    chunks = chunk_text(text, chunk_size_words=120, overlap_words=20)
    # embed chunks in batches
    embeddings = []
    try:
        embeddings = embed_texts(chunks)
    except Exception as e:
        return jsonify({"error":"Embedding error","detail":str(e)}), 502
    # save each chunk+embedding to DB with metadata (store parent id/slug if provided)
    parent = payload.get("id")
    for i, c in enumerate(chunks):
        md = dict(metadata)
        if parent:
            md["_parent"] = parent
        md["_chunk_index"] = i
        save_doc(c, embeddings[i], md)
    return jsonify({"status":"ok","chunks_indexed": len(chunks)}), 200

# ---------- search + RAG query endpoint ----------
@app.route("/rag-query", methods=["POST"])
def rag_query():
    """
    POST JSON:
      {"query":"...", "top_k": 3, "use_llm": true/false}
    Returns top_k matching chunks and, if use_llm true, LLM-generated answer.
    """
    if not request.is_json:
        return jsonify({"error":"JSON body required"}), 400
    payload = request.get_json()
    query = payload.get("query", "").strip()
    if not query:
        return jsonify({"error":"'query' required"}), 400
    top_k = int(payload.get("top_k", 3))
    use_llm = bool(payload.get("use_llm", True))

    # embed query
    try:
        q_emb = embed_texts([query])[0]
    except Exception as e:
        return jsonify({"error":"Embedding error","detail":str(e)}), 502

    # load all embeddings from DB and compute similarity (note: O(n) scan)
    rows = load_all_embeddings()
    scored = []
    for r in rows:
        score = cosine_similarity(q_emb, r["embedding"])
        scored.append({"id": r["id"], "text": r["text"], "score": score, "metadata": r["metadata"]})
    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:top_k]

    result = {"query": query, "top_matches": top}

    if use_llm:
        # assemble context - join top texts (truncate to MAX_CONTEXT_CHARS)
        context_parts = []
        total = 0
        for t in top:
            txt = t["text"]
            if total + len(txt) > MAX_CONTEXT_CHARS:
                # truncate last piece if needed
                remain = MAX_CONTEXT_CHARS - total
                if remain <= 0:
                    break
                txt = txt[:remain]
            context_parts.append(txt)
            total += len(txt)
        context = "\n\n---\n\n".join(context_parts)

        system_prompt = (
            "You are a helpful assistant. Use ONLY the context below to answer the user's question. "
            "If the answer isn't contained in the context, say you don't know. Be concise."
        )
        user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"

        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            ai_resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role":"system", "content": system_prompt},
                    {"role":"user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0.0,
                timeout=30
            )
            answer = ai_resp.choices[0].message.content.strip()
            result["answer"] = answer
        except Exception as e:
            return jsonify({"error":"OpenAI API error","detail":str(e)}), 502

    return jsonify(result), 200


# ========================================
# 12. /rag-docs – List all stored RAG documents
# ========================================
@app.route("/rag-docs", methods=["GET"])
def rag_docs():
    """
    Return the list of indexed chunks from the SQLite vector store.
    Query params:
      - include_embeddings=true  -> returns the full embedding lists (large!)
    """
    include_embeddings = request.args.get("include_embeddings", "false").lower() in ("1", "true", "yes")

    conn = sqlite3.connect(VECTOR_DB)
    cur = conn.cursor()
    cur.execute("SELECT id, text, embedding, metadata FROM docs ORDER BY id ASC")
    rows = cur.fetchall()
    conn.close()

    docs = []
    for r in rows:
        emb_json = r[2]
        # avoid loading full embeddings unless explicitly requested
        embedding = json.loads(emb_json) if (emb_json and include_embeddings) else None
        emb_len = len(json.loads(emb_json)) if emb_json else 0

        metadata = {}
        try:
            metadata = json.loads(r[3]) if r[3] else {}
        except Exception:
            metadata = {"_raw_metadata": r[3]}

        docs.append({
            "id": r[0],
            "text": r[1],
            "embedding_length": emb_len,
            "embedding": embedding,           # None by default (to keep response small)
            "metadata": metadata
        })

    return jsonify({
        "count": len(docs),
        "chunks": docs
    }), 200

# ========================================
# 13. /rag-delete – Remove a stored chunk by id
# ========================================
@app.route("/rag-delete", methods=["POST"])
def rag_delete():
    data = request.json
    if not data or "id" not in data:
        return jsonify({"error": "Provide 'id' to delete"}), 400

    chunk_id = data["id"]

    try:
        conn = sqlite3.connect("vector_store.db")
        cur = conn.cursor()

        # Check if exists
        cur.execute("SELECT id FROM docs WHERE id = ?", (chunk_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return jsonify({"error": "No such chunk ID"}), 404

        # Delete
        cur.execute("DELETE FROM docs WHERE id = ?", (chunk_id,))
        conn.commit()
        conn.close()

        return jsonify({"status": "deleted", "id": chunk_id}), 200

    except Exception as e:
        return jsonify({"error": "delete failed", "detail": str(e)}), 500

# ========================================
# Main entry point
# ========================================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)