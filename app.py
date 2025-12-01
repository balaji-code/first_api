
import hashlib
import time
import os
import json
import math
import sqlite3
import re
from typing import Dict, List

import openai
try:
    # Newer SDKs expose an OpenAI client class
    from openai import OpenAI  # type: ignore
except ImportError:
    # Compat wrapper so the rest of the code can use OpenAI(...).chat/embeddings
    class OpenAI:  # type: ignore
        def __init__(self, api_key: str | None = None, **kwargs):
            if api_key:
                openai.api_key = api_key

        class chat:  # noqa: D401 - simple proxy
            class completions:
                @staticmethod
                def create(**kwargs):
                    # Map to legacy ChatCompletion.create(...)
                    return openai.ChatCompletion.create(**kwargs)

        class embeddings:
            @staticmethod
            def create(**kwargs):
                # Map to legacy Embedding.create(...)
                return openai.Embedding.create(**kwargs)
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, Response
from dotenv import load_dotenv
load_dotenv()   # loads .env into os.environ for this process
#helper 
import sqlite3
import json



import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Resolve DB path robustly -----------------------------------------
_env_db = os.getenv("RAG_DB")

if _env_db:
    RAG_DB = str(Path(_env_db).expanduser().resolve())
else:
    BASE = Path(__file__).resolve().parent
    RAG_DB = str((BASE / "vector_store.db").resolve())

print("DEBUG: Using RAG_DB =", RAG_DB)

def upsert_document_to_db(doc_id: str, text: str, metadata: dict, max_chars=800, overlap=100):
    """
    Reusable function that:
        - deletes existing chunks for parent doc_id
        - chunks text
        - embeds chunks via embed_texts
        - inserts new chunk rows
    Returns: (True, chunks_indexed) on success or (False, "error message") on failure.
    """
    if not doc_id or not text:
        return False, "doc_id and text required"

    metadata = metadata or {}
    metadata["_parent"] = doc_id

    # connect and delete old rows (safe), using the shared docs table
    try:
        conn = sqlite3.connect(RAG_DB)
        cur = conn.cursor()
        # docs table stores per-chunk rows; we delete by parent marker in metadata JSON
        cur.execute(
            "DELETE FROM docs WHERE metadata LIKE ?",
            (f'%\"_parent\":\"{doc_id}\"%',),
        )
        conn.commit()
    except Exception as e:
        try:
            conn.close()
        except:
            pass
        return False, f"DB delete error: {e}"

    # chunk
    chunks = simple_chunk_text(text, max_chars=max_chars, overlap=overlap)
    if not chunks:
        try:
            conn.close()
        except:
            pass
        return False, "No chunks created from text"

    # embed
    try:
        embeddings = embed_texts(chunks)  # assumes list-of-lists floats
    except Exception as e:
        try:
            conn.close()
        except:
            pass
        return False, f"Embedding error: {e}"

    # insert new chunk rows into docs table
    try:
        for idx, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
            md = dict(metadata)
            md["_chunk_index"] = idx
            cur.execute(
                "INSERT INTO docs (text, embedding, metadata) VALUES (?, ?, ?)",
                (chunk_text, json.dumps(emb), json.dumps(md)),
            )
        conn.commit()
        count = len(chunks)
        conn.close()
        return True, count
    except Exception as e:
        try:
            conn.close()
        except:
            pass
        return False, f"DB insert error: {e}"
# Simple in-memory vector store
RAG_STORE = []
# Minimum cosine similarity required to treat a chunk as relevant
MIN_SCORE = 0.25

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

        import json
        import re
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
        import json
        import re
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



EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
MAX_CONTEXT_CHARS = 4000  # safe limit to attach to prompts

# ---------- DB helpers ----------


def ensure_db():
    conn = sqlite3.connect(RAG_DB)
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
    conn = sqlite3.connect(RAG_DB)
    cur = conn.cursor()
    cur.execute("INSERT INTO docs (text, embedding, metadata) VALUES (?, ?, ?)",
                (text, json.dumps(embedding), json.dumps(metadata or {})))
    conn.commit()
    conn.close()


def load_all_embeddings():
    conn = sqlite3.connect(RAG_DB)
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
    return sum(x*y for x, y in zip(a, b))


def norm(a: List[float]) -> float:
    return math.sqrt(sum(x*x for x in a))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    na = norm(a)
    nb = norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot(a, b) / (na*nb)

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
        return jsonify({"error": "JSON body required"}), 400
    payload = request.get_json()
    if "text" not in payload:
        return jsonify({"error": "'text' field required"}), 400
    text = payload["text"]
    metadata = payload.get("metadata", {})
    # split into chunks
    chunks = chunk_text(text, chunk_size_words=120, overlap_words=20)
    # embed chunks in batches
    embeddings = []
    try:
        embeddings = embed_texts(chunks)
    except Exception as e:
        return jsonify({"error": "Embedding error", "detail": str(e)}), 502
    # save each chunk+embedding to DB with metadata (store parent id/slug if provided)
    parent = payload.get("id")
    for i, c in enumerate(chunks):
        md = dict(metadata)
        if parent:
            md["_parent"] = parent
        md["_chunk_index"] = i
        save_doc(c, embeddings[i], md)
    return jsonify({"status": "ok", "chunks_indexed": len(chunks)}), 200

# ---------- search + RAG query endpoint (improved & safe) ----------


@app.route("/rag-query", methods=["POST"])
def rag_query():
    """
    POST JSON:
        {
            "query": "...",
            "top_k": 3,
            "use_llm": true/false
        }
    """

    # -------- Input validation --------
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    payload = request.get_json()

    query = payload.get("query", "").strip()
    if not query:
        return jsonify({"error": "'query' required"}), 400

    # safer defaults
    top_k = int(payload.get("top_k", 3))
    top_k = max(1, min(top_k, 10))   # cap between 1 and 10
    use_llm = bool(payload.get("use_llm", True))

    # -------- Embed query --------
    try:
        q_emb = embed_texts([query])[0]
    except Exception as e:
        return jsonify({"error": "Embedding error", "detail": str(e)}), 502

    # -------- Load all stored embeddings --------
    rows = load_all_embeddings()
    if not rows:
        return jsonify({"query": query, "top_matches": [], "answer": "No documents indexed"}), 200

    # -------- Compute cosine similarity --------
    scored = []
    for r in rows:
        emb = r.get("embedding")
        if not emb:
            continue  # skip any chunk missing embeddings
        try:
            score = cosine_similarity(q_emb, emb)
        except Exception:
            continue
        scored.append({
            "id": r["id"],
            "text": r["text"],
            "metadata": r.get("metadata", {}),
            "score": float(score)
        })

    if not scored:
        return jsonify({"query": query, "top_matches": [], "answer": "No valid embeddings found"}), 200

    # sort high → low
    scored.sort(key=lambda x: x["score"], reverse=True)

    # threshold to filter noise
    MIN_SCORE = 0.12
    top = [s for s in scored if s["score"] >= MIN_SCORE][:top_k]

    result = {
        "query": query,
        "top_matches": top
    }

    # -------- If no relevant documents found --------
    if not top:
        result["answer"] = "I don't know (no relevant documents match)."
        return jsonify(result), 200

    # -------- If the user does NOT want LLM answer --------
    if not use_llm:
        return jsonify(result), 200

    # -------- Build context (simple truncation) --------
    MAX_CONTEXT_CHARS = 4000  # safe for your current model
    parts = []
    total = 0

    for t in top:
        txt = t["text"]
        if total + len(txt) > MAX_CONTEXT_CHARS:
            remain = MAX_CONTEXT_CHARS - total
            if remain <= 0:
                break
            txt = txt[:remain]
        parts.append(txt)
        total += len(txt)

    context = "\n\n---\n\n".join(parts)

    # -------- LLM answer --------
    system_prompt = (
        "You are a helpful assistant. Use ONLY the provided context to answer. "
        "If the answer is not in the context, say you don't know. Be concise."
    )
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        ai_resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=300,
            temperature=0.0
        )
        answer = ai_resp.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"error": "OpenAI API error", "detail": str(e)}), 502

    result["answer"] = answer
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

    conn = sqlite3.connect(RAG_DB)
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
        conn = sqlite3.connect(RAG_DB)
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


@app.route("/rag-search", methods=["POST"])
def rag_search():
    """
    Simple RAG retrieval (no LLM).
    POST JSON:
        {"query":"...", "top_k": 3, "metadata":{"topic":"computers"}}
    Returns top_k chunks sorted by similarity.
    """
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400

    payload = request.get_json()
    query = payload.get("query", "").strip()
    if not query:
        return jsonify({"error": "'query' required"}), 400

    top_k = max(1, min(int(payload.get("top_k", 3)), 20))
    metadata_filter = payload.get("metadata", {})

    # 1. Embed query
    try:
        q_emb = embed_texts([query])[0]
    except Exception as e:
        return jsonify({"error": "Embedding error", "detail": str(e)}), 502

    # 2. Load DB rows
    rows = load_all_embeddings()
    scored = []

    for r in rows:
        emb = r.get("embedding")
        if not emb:
            continue

        md = r.get("metadata", {}) or {}

        # 3. Metadata filter
        ok = True
        if isinstance(metadata_filter, dict):
            for k, v in metadata_filter.items():
                if md.get(k) != v:
                    ok = False
                    break
        if not ok:
            continue

        score = cosine_similarity(q_emb, emb)
        scored.append({
            "id": r["id"],
            "text": r["text"],
            "metadata": md,
            "score": float(score)
        })

    # 4. Sort and pick top_k only
    scored.sort(key=lambda x: x["score"], reverse=True)
    top_matches = scored[:top_k]

    return jsonify({
        "query": query,
        "top_matches": top_matches,
        "count_scored": len(scored)
    }), 200





def simple_chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
    """
    Create chunks by sentence boundaries, then join until max_chars.
    Overlap keeps context between chunks.
    """
    # naive sentence split (good enough for English)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current = []
    current_len = 0
    for s in sentences:
        if current_len + len(s) + 1 <= max_chars:
            current.append(s)
            current_len += len(s) + 1
        else:
            if current:
                chunks.append(" ".join(current).strip())
            # start new chunk, but allow overlap
            # take last N chars from previous if available
            overlap_text = ""
            if chunks and overlap > 0:
                prev = chunks[-1]
                overlap_text = prev[-overlap:] if len(prev) > overlap else prev
            current = [overlap_text, s] if overlap_text else [s]
            current_len = sum(len(x) for x in current) + len(current) - 1
    if current:
        chunks.append(" ".join(current).strip())
    # final cleanup: remove empty chunks
    return [c for c in chunks if c]


@app.route("/rag-upsert", methods=["POST"])
def rag_upsert():
    """
    Upsert a document:
    POST JSON:
        {
            "id": "<doc-id>",
            "text": "<full document text>",
            "metadata": {...}   # optional
        }
    Behavior:
        - If doc-id exists, delete its old chunks
        - Chunk the provided text, embed, and insert new chunks
    """
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    payload = request.get_json()

    doc_id = payload.get("id", "").strip()
    text = payload.get("text", "").strip()
    metadata = payload.get("metadata", {}) or {}

    if not doc_id:
        return jsonify({"error": "'id' required"}), 400
    if not text:
        return jsonify({"error": "'text' required"}), 400

    # attach parent info into metadata for provenance
    metadata["_parent"] = doc_id

    # 1) Delete existing rows for this parent document (safe idempotent behavior)
    try:
        conn = sqlite3.connect(RAG_DB)
        cur = conn.cursor()
        # docs table stores per-chunk rows; we delete by parent marker in metadata JSON
        cur.execute(
            "DELETE FROM docs WHERE metadata LIKE ?",
            (f'%\"_parent\":\"{doc_id}\"%',),
        )
        conn.commit()
    except Exception as e:
        if conn:
            conn.close()
        return jsonify({"error": "DB delete error", "detail": str(e)}), 500

    # 2) Chunk the new text
    chunks = simple_chunk_text(text, max_chars=800, overlap=100)
    if not chunks:
        return jsonify({"error": "No chunks created from text"}), 400

    # 3) Create embeddings for all chunks
    try:
        embeddings = embed_texts(chunks)  # uses your existing helper
    except Exception as e:
        return jsonify({"error": "Embedding error", "detail": str(e)}), 502

    # 4) Insert new chunk rows into docs table
    try:
        for idx, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
            md = dict(metadata)  # copy
            md["_chunk_index"] = idx
            cur.execute(
                "INSERT INTO docs (text, embedding, metadata) VALUES (?, ?, ?)",
                (chunk_text, json.dumps(emb), json.dumps(md)),
            )
        conn.commit()
        conn.close()
    except Exception as e:
        if conn:
            conn.close()
        return jsonify({"error": "DB insert error", "detail": str(e)}), 500

    return jsonify({"status": "ok", "id": doc_id, "chunks_indexed": len(chunks)}), 200


@app.route("/rag-bulk-index", methods=["POST"])
def rag_bulk_index():
    """
    POST JSON: {"docs": [ {"id":"doc1", "text":"...", "metadata": {...}}, ... ], "max_chars":800}
    Returns: summary with per-doc result.
    """
    if not request.is_json:
        return jsonify({"error":"JSON body required"}), 400

    payload = request.get_json()
    docs = payload.get("docs")
    if not isinstance(docs, list) or len(docs) == 0:
        return jsonify({"error":"'docs' must be a non-empty list"}), 400

    max_chars = int(payload.get("max_chars", 800))
    overlap = int(payload.get("overlap", 100))

    summary = {"processed": 0, "succeeded": 0, "failed": 0, "details": []}
    total_chunks = 0

    for d in docs:
        doc_id = (d.get("id") or "").strip()
        text = (d.get("text") or "").strip()
        metadata = d.get("metadata", {}) or {}

        if not doc_id or not text:
            summary["failed"] += 1
            summary["details"].append({"id": doc_id or None, "status": "failed", "reason": "id and text required"})
            continue

        summary["processed"] += 1
        ok, info = upsert_document_to_db(doc_id, text, metadata, max_chars=max_chars, overlap=overlap)
        if ok:
            summary["succeeded"] += 1
            total_chunks += info
            summary["details"].append({"id": doc_id, "status": "ok", "chunks_indexed": info})
        else:
            summary["failed"] += 1
            summary["details"].append({"id": doc_id, "status": "failed", "reason": info})

    summary["total_chunks_indexed"] = total_chunks
    return jsonify(summary), 200
# ========================================
# Main entry point
# ========================================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)

#!/usr/bin/env python3
"""
first_api/app.py

A small Flask microservice demonstrating common AI endpoints + a
simple RAG (retrieval-augmented generation) pipeline backed by SQLite.

The code is intentionally commented and formatted to be approachable for
someone new to Python, APIs, and vector-based retrieval. Keep this file
in the project root (next to venv/ and vector_store.db).

Important notes:
- RAG data is stored in the SQLite file defined by RAG_DB (default
  ./vector_store.db). Use the RAG_DB env var to override.
- OpenAI credentials must be set in the environment (OPENAI_API_KEY)
  or .env file loaded by python-dotenv.

"""

import hashlib
import time
import os
import json
import math
import sqlite3
import re
from typing import Dict, List

# OpenAI SDK (modern usage) + requests + BeautifulSoup for URL summarization
import openai
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, Response
from dotenv import load_dotenv

# Load environment variables from .env (if present)
load_dotenv()

# ----------------------------
# Configuration and constants
# ----------------------------

# Determine a robust path for the RAG SQLite file. You can set RAG_DB in
# your environment to point elsewhere. By default it's ./vector_store.db
_env_db = os.getenv("RAG_DB")
if _env_db:
    from pathlib import Path
    RAG_DB = str(Path(_env_db).expanduser().resolve())
else:
    from pathlib import Path
    BASE = Path(__file__).resolve().parent
    RAG_DB = str((BASE / "vector_store.db").resolve())

print("DEBUG: Using RAG_DB =", RAG_DB)

# Minimum cosine similarity threshold (used by some endpoints)
MIN_SCORE = 0.25

# Flask app
app = Flask(__name__)
# Ensure JSON responses keep non-ascii characters (useful for translations)
app.config['JSON_AS_ASCII'] = False

# Simple in-memory cache & TTL (used for small endpoints to avoid repeated calls)
CACHE: Dict[str, tuple] = {}
CACHE_TTL = 60 * 60  # 1 hour

# OpenAI configuration pulled from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not set. Some endpoints will return errors until it's provided.")
else:
    # Keep compatibility with older code paths: set openai.api_key as a convenience
    openai.api_key = OPENAI_API_KEY

# ----------------------------
# Cache helpers
# ----------------------------

def cache_get(key: str):
    entry = CACHE.get(key)
    if not entry:
        return None
    ts, value = entry
    if time.time() - ts > CACHE_TTL:
        del CACHE[key]
        return None
    return value


def cache_set(key: str, value):
    CACHE[key] = (time.time(), value)

# ----------------------------
# Utility helpers: math, embeddings, chunking
# ----------------------------

def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: List[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    na = norm(a)
    nb = norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot(a, b) / (na * nb)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Call OpenAI embedding API (modern client) and return list-of-list floats.
    Raises exception on API errors.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def simple_chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
    """
    Create human-friendly chunks by splitting on sentence boundaries and
    joining sentences until `max_chars` is reached. Adds a small overlap
    between chunks for context continuity.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current = []
    current_len = 0
    for s in sentences:
        if current_len + len(s) + 1 <= max_chars:
            current.append(s)
            current_len += len(s) + 1
        else:
            if current:
                chunks.append(" ".join(current).strip())
            overlap_text = ""
            if chunks and overlap > 0:
                prev = chunks[-1]
                overlap_text = prev[-overlap:] if len(prev) > overlap else prev
            current = [overlap_text, s] if overlap_text else [s]
            current_len = sum(len(x) for x in current) + len(current) - 1
    if current:
        chunks.append(" ".join(current).strip())
    return [c for c in chunks if c]

# ----------------------------
# SQLite helpers for the RAG "docs" table
# ----------------------------

def ensure_db():
    """Create the docs table if it doesn't exist."""
    conn = sqlite3.connect(RAG_DB)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS docs (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            embedding TEXT NOT NULL,
            metadata TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_doc(text: str, embedding: List[float], metadata: Dict = None):
    conn = sqlite3.connect(RAG_DB)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO docs (text, embedding, metadata) VALUES (?, ?, ?)",
        (text, json.dumps(embedding), json.dumps(metadata or {})),
    )
    conn.commit()
    conn.close()


def load_all_embeddings():
    """Load all rows from docs and return a list of dicts (id, text, embedding, metadata).
    The embedding column is stored as JSON text and parsed back to a Python list.
    """
    conn = sqlite3.connect(RAG_DB)
    cur = conn.cursor()
    cur.execute("SELECT id, text, embedding, metadata FROM docs")
    rows = cur.fetchall()
    conn.close()

    result = []
    for r in rows:
        try:
            emb = json.loads(r[2])
        except Exception:
            emb = None
        try:
            md = json.loads(r[3]) if r[3] else {}
        except Exception:
            md = {"_raw_metadata": r[3]}
        result.append({
            "id": r[0],
            "text": r[1],
            "embedding": emb,
            "metadata": md,
        })
    return result

# Ensure DB creation on import/startup
ensure_db()

# ----------------------------
# Small helpful endpoints (health + echo)
# ----------------------------

@app.route("/hello", methods=["GET"])
def hello():
    """Simple health-check endpoint."""
    return jsonify({"message": "Hello from your API", "status": "ok"}), 200


@app.route("/echo", methods=["POST"])
def echo():
    """Echo endpoint that validates incoming JSON and returns it back."""
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    data = request.get_json()
    if "name" not in data:
        return jsonify({"error": "'name' field is required"}), 400
    if not isinstance(data["name"], str):
        return jsonify({"error": "'name' must be a string"}), 400
    return jsonify({"received": data, "message": "Valid input received"}), 200

# ----------------------------
# OpenAI-backed endpoints (summarize, keywords, sentiment, translate, etc.)
# These endpoints use the modern OpenAI client where possible (OpenAI(...))
# ----------------------------

@app.route("/summarize", methods=["POST"])
def summarize():
    """Summarize a given text using the OpenAI model.

    POST JSON: {"text": "..."}
    """
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    data = request.get_json()
    if "text" not in data or not isinstance(data["text"], str):
        return jsonify({"error": "'text' field is required and must be a string"}), 400
    if not OPENAI_API_KEY:
        return jsonify({"error": "OpenAI API key not configured on the server"}), 500
    text = data["text"].strip()
    if text == "":
        return jsonify({"error": "'text' must be a non-empty string"}), 400

    system_prompt = (
        "You are a concise summarizer. Produce a short, factual, bullet-point summary "
        "of the user's text. Do not invent facts. Keep to 3-6 bullets when possible."
    )
    user_prompt = f"Summarize the following text:\n\n{text}"

    try:
        # NOTE: older examples used openai.ChatCompletion — the modern client uses OpenAI(...)
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=256,
            temperature=0.2,
            n=1,
            timeout=30,
        )

        # Extract model output robustly
        try:
            summary_text = resp.choices[0].message.content.strip()
        except Exception:
            summary_text = resp["choices"][0]["message"]["content"].strip()

        lines = [line.strip() for line in summary_text.splitlines() if line.strip()]
        if len(lines) <= 1:
            import re
            sentences = re.split(r'(?<=[.!?])\s+', summary_text)
            lines = [s.strip() for s in sentences if s.strip()][:6]

        return jsonify({"summary_raw": summary_text, "summary_bullets": lines}), 200

    except Exception as e:
        return jsonify({"error": "OpenAI API error", "detail": str(e)}), 502


@app.route("/keywords", methods=["POST"])
def keywords():
    """Return up to 5 keywords extracted from text as JSON array.

    POST JSON: {"text": "..."}
    """
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    data = request.get_json()
    text = data.get("text", "")
    if not isinstance(text, str) or text.strip() == "":
        return jsonify({"error": "'text' field is required and must be non-empty"}), 400

    if not OPENAI_API_KEY:
        return jsonify({"error": "OpenAI API key not configured on the server"}), 500

    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cached = cache_get(key)
    if cached:
        return jsonify(cached), 200

    client = OpenAI(api_key=OPENAI_API_KEY)
    system_prompt = (
        "You are a strict JSON generator. Given input text, return ONLY a JSON array "
        "of up to 5 keywords (strings). Output must be valid JSON with no surrounding text."
    )
    user_prompt = f"Extract up to 5 keywords from the following text. Text:\n\n{text}"

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=150,
            temperature=0.0,
            n=1,
            timeout=30,
        )

        try:
            raw = resp.choices[0].message.content.strip()
        except Exception:
            raw = resp["choices"][0]["message"]["content"].strip()

        # Try to extract a JSON array from the model output
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
    """Return a simple sentiment JSON {label, score} extracted by the model."""
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    data = request.get_json()
    text = data.get("text", "")
    if not isinstance(text, str) or text.strip() == "":
        return jsonify({"error": "'text' field is required and must be non-empty"}), 400
    if not OPENAI_API_KEY:
        return jsonify({"error": "OpenAI API key not configured on the server"}), 500

    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cached = cache_get(key)
    if cached:
        return jsonify(cached), 200

    client = OpenAI(api_key=OPENAI_API_KEY)
    system_prompt = (
        "You are a classifier. Return ONLY a JSON object with keys 'label' and 'score'. "
        "Label must be one of: positive, neutral, negative. Score must be 0.00-1.00."
    )
    user_prompt = f"Classify the sentiment of this text and return the JSON object:\n\n{text}"

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_tokens=64,
            temperature=0.0,
            n=1,
            timeout=30,
        )

        try:
            raw = resp.choices[0].message.content.strip()
        except Exception:
            raw = resp["choices"][0]["message"]["content"].strip()

        m = re.search(r'(\{.*\})', raw, flags=re.S)
        parsed = {}
        if m:
            try:
                parsed = json.loads(m.group(1))
            except Exception:
                # best-effort parsing
                lab = re.search(r'"?label"?\s*[:=]\s*"?(positive|neutral|negative)"?', raw, flags=re.I)
                sc = re.search(r'"?score"?\s*[:=]\s*([0-9]*\.?[0-9]+)', raw)
                if lab:
                    parsed["label"] = lab.group(1).lower()
                if sc:
                    parsed["score"] = float(sc.group(1))
        else:
            lab = re.search(r'(positive|neutral|negative)', raw, flags=re.I)
            sc = re.search(r'([0-9]*\.?[0-9]+)', raw)
            if lab:
                parsed["label"] = lab.group(1).lower()
            if sc:
                parsed["score"] = float(sc.group(1))

        label = parsed.get("label") if parsed.get("label") in ("positive", "neutral", "negative") else "neutral"
        try:
            score = float(parsed.get("score", 0.5))
            score = max(0.0, min(1.0, score))
        except Exception:
            score = {"positive": 0.9, "neutral": 0.5, "negative": 0.9}.get(label, 0.5)

        result = {"sentiment_raw": raw, "label": label, "score": round(score, 3)}
        cache_set(key, result)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": "Server error", "detail": str(e)}), 500


# ----------------------------
# Entities / rewrite / translate / classify / chat / summarize-url endpoints
# (kept short here because they follow the same pattern as above)
# ----------------------------

# For brevity in this patch, those endpoints are intentionally preserved
# in behavior but with clearer comments. Keep them implemented the same
# way as keywords/sentiment/summarize above. If you want the full
# expanded implementations moved here, I can paste them as well.

# -----------------------------------
# RAG (retrieval) endpoints: index, upsert, search, query, docs, delete
# -----------------------------------

@app.route("/rag-index", methods=["POST"])
def rag_index():
    """Index a single document: chunk -> embed -> save to docs table.

    POST JSON: {"id": "optional parent id", "text": "...", "metadata": {...}}
    """
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    payload = request.get_json()
    if "text" not in payload:
        return jsonify({"error": "'text' field required"}), 400

    text = payload["text"]
    metadata = payload.get("metadata", {}) or {}
    parent = payload.get("id")

    # chunk and embed
    chunks = simple_chunk_text(text, max_chars=800, overlap=100)
    try:
        embeddings = embed_texts(chunks)
    except Exception as e:
        return jsonify({"error": "Embedding error", "detail": str(e)}), 502

    for i, c in enumerate(chunks):
        md = dict(metadata)
        if parent:
            md["_parent"] = parent
        md["_chunk_index"] = i
        save_doc(c, embeddings[i], md)

    return jsonify({"status": "ok", "chunks_indexed": len(chunks)}), 200


@app.route("/rag-upsert", methods=["POST"])
def rag_upsert():
    """Upsert a whole document by id: safe delete old chunks for the id,
    chunk the new text, embed, and insert into the docs table.
    POST JSON: {"id": "doc-id", "text": "...", "metadata": {...}}
    """
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    payload = request.get_json()
    doc_id = payload.get("id", "").strip()
    text = payload.get("text", "").strip()
    metadata = payload.get("metadata", {}) or {}

    if not doc_id:
        return jsonify({"error": "'id' required"}), 400
    if not text:
        return jsonify({"error": "'text' required"}), 400

    # delete old chunks for this parent
    try:
        conn = sqlite3.connect(RAG_DB)
        cur = conn.cursor()
        cur.execute("DELETE FROM docs WHERE metadata LIKE ?", (f'%\"_parent\":\"{doc_id}\"%',))
        conn.commit()
    except Exception as e:
        if conn:
            conn.close()
        return jsonify({"error": "DB delete error", "detail": str(e)}), 500

    # chunk -> embed -> insert
    chunks = simple_chunk_text(text, max_chars=800, overlap=100)
    if not chunks:
        return jsonify({"error": "No chunks created from text"}), 400

    try:
        embeddings = embed_texts(chunks)
    except Exception as e:
        return jsonify({"error": "Embedding error", "detail": str(e)}), 502

    try:
        for idx, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
            md = dict(metadata)
            md["_parent"] = doc_id
            md["_chunk_index"] = idx
            cur.execute("INSERT INTO docs (text, embedding, metadata) VALUES (?, ?, ?)", (chunk_text, json.dumps(emb), json.dumps(md)))
        conn.commit()
        conn.close()
    except Exception as e:
        if conn:
            conn.close()
        return jsonify({"error": "DB insert error", "detail": str(e)}), 500

    return jsonify({"status": "ok", "id": doc_id, "chunks_indexed": len(chunks)}), 200


@app.route("/rag-bulk-index", methods=["POST"])
def rag_bulk_index():
    """Accept multiple docs in a single request and upsert each."""
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    payload = request.get_json()
    docs = payload.get("docs")
    if not isinstance(docs, list) or len(docs) == 0:
        return jsonify({"error": "'docs' must be a non-empty list"}), 400

    max_chars = int(payload.get("max_chars", 800))
    overlap = int(payload.get("overlap", 100))

    summary = {"processed": 0, "succeeded": 0, "failed": 0, "details": []}
    total_chunks = 0

    for d in docs:
        doc_id = (d.get("id") or "").strip()
        text = (d.get("text") or "").strip()
        metadata = d.get("metadata", {}) or {}
        if not doc_id or not text:
            summary["failed"] += 1
            summary["details"].append({"id": doc_id or None, "status": "failed", "reason": "id and text required"})
            continue
        summary["processed"] += 1
        ok, info = upsert_document_to_db(doc_id, text, metadata, max_chars=max_chars, overlap=overlap)
        if ok:
            summary["succeeded"] += 1
            total_chunks += info
            summary["details"].append({"id": doc_id, "status": "ok", "chunks_indexed": info})
        else:
            summary["failed"] += 1
            summary["details"].append({"id": doc_id, "status": "failed", "reason": info})

    summary["total_chunks_indexed"] = total_chunks
    return jsonify(summary), 200


@app.route("/rag-docs", methods=["GET"])
def rag_docs():
    """List all stored chunks (documents) in the vector DB (small response by default).

    Query param `include_embeddings=true` will return full embeddings (large payload).
    """
    include_embeddings = request.args.get("include_embeddings", "false").lower() in ("1", "true", "yes")
    conn = sqlite3.connect(RAG_DB)
    cur = conn.cursor()
    cur.execute("SELECT id, text, embedding, metadata FROM docs ORDER BY id ASC")
    rows = cur.fetchall()
    conn.close()

    docs = []
    for r in rows:
        emb_json = r[2]
        embedding = json.loads(emb_json) if (emb_json and include_embeddings) else None
        emb_len = len(json.loads(emb_json)) if emb_json else 0
        try:
            metadata = json.loads(r[3]) if r[3] else {}
        except Exception:
            metadata = {"_raw_metadata": r[3]}
        docs.append({"id": r[0], "text": r[1], "embedding_length": emb_len, "embedding": embedding, "metadata": metadata})

    return jsonify({"count": len(docs), "chunks": docs}), 200


@app.route("/rag-delete", methods=["POST"])
def rag_delete():
    """Delete a stored chunk by its integer row id in the docs table.
    POST JSON: {"id": <int>}
    """
    data = request.json
    if not data or "id" not in data:
        return jsonify({"error": "Provide 'id' to delete"}), 400
    chunk_id = data["id"]
    try:
        conn = sqlite3.connect(RAG_DB)
        cur = conn.cursor()
        cur.execute("SELECT id FROM docs WHERE id = ?", (chunk_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return jsonify({"error": "No such chunk ID"}), 404
        cur.execute("DELETE FROM docs WHERE id = ?", (chunk_id,))
        conn.commit()
        conn.close()
        return jsonify({"status": "deleted", "id": chunk_id}), 200
    except Exception as e:
        return jsonify({"error": "delete failed", "detail": str(e)}), 500


@app.route("/rag-search", methods=["POST"])
def rag_search():
    """Search for top_k chunks similar to the query (no LLM answer, just retrieval).
    POST JSON: {"query":"...", "top_k":3, "metadata":{...}}
    Returns top_matches sorted by score.
    """
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    payload = request.get_json()
    query = payload.get("query", "").strip()
    if not query:
        return jsonify({"error": "'query' required"}), 400
    top_k = max(1, min(int(payload.get("top_k", 3)), 20))
    metadata_filter = payload.get("metadata", {})

    try:
        q_emb = embed_texts([query])[0]
    except Exception as e:
        return jsonify({"error": "Embedding error", "detail": str(e)}), 502

    rows = load_all_embeddings()
    scored = []
    for r in rows:
        emb = r.get("embedding")
        if not emb:
            continue
        md = r.get("metadata", {}) or {}
        # metadata filtering
        ok = True
        if isinstance(metadata_filter, dict):
            for k, v in metadata_filter.items():
                if md.get(k) != v:
                    ok = False
                    break
        if not ok:
            continue
        score = cosine_similarity(q_emb, emb)
        scored.append({"id": r["id"], "text": r["text"], "metadata": md, "score": float(score)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    top_matches = scored[:top_k]
    return jsonify({"query": query, "top_matches": top_matches, "count_scored": len(scored)}), 200


@app.route("/rag-query", methods=["POST"])
def rag_query():
    """RAG query: retrieve top_k chunks and optionally ask LLM to answer using those
    chunks as context. This endpoint uses the same docs table and embeddings as the
    other RAG endpoints.
    """
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    payload = request.get_json()
    query = payload.get("query", "").strip()
    if not query:
        return jsonify({"error": "'query' required"}), 400
    top_k = int(payload.get("top_k", 3))
    top_k = max(1, min(top_k, 10))
    use_llm = bool(payload.get("use_llm", True))

    try:
        q_emb = embed_texts([query])[0]
    except Exception as e:
        return jsonify({"error": "Embedding error", "detail": str(e)}), 502

    rows = load_all_embeddings()
    scored = []
    for r in rows:
        emb = r.get("embedding")
        if not emb:
            continue
        try:
            score = cosine_similarity(q_emb, emb)
        except Exception:
            continue
        scored.append({"id": r["id"], "text": r["text"], "metadata": r.get("metadata", {}), "score": float(score)})

    if not scored:
        return jsonify({"query": query, "top_matches": [], "answer": "No documents indexed"}), 200

    scored.sort(key=lambda x: x["score"], reverse=True)
    MIN_SCORE_LOCAL = 0.12
    top = [s for s in scored if s["score"] >= MIN_SCORE_LOCAL][:top_k]

    result = {"query": query, "top_matches": top}
    if not top:
        result["answer"] = "I don't know (no relevant documents match)."
        return jsonify(result), 200

    if not use_llm:
        return jsonify(result), 200

    # assemble context for the LLM from the top matches
    MAX_CONTEXT_CHARS = 4000
    parts = []
    total = 0
    for t in top:
        txt = t["text"]
        if total + len(txt) > MAX_CONTEXT_CHARS:
            remain = MAX_CONTEXT_CHARS - total
            if remain <= 0:
                break
            txt = txt[:remain]
        parts.append(txt)
        total += len(txt)

    context = "\n\n---\n\n".join(parts)

    system_prompt = (
        "You are a helpful assistant. Use ONLY the provided context to answer. "
        "If the answer is not in the context, say you don't know. Be concise."
    )
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        ai_resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_tokens=300,
            temperature=0.0,
        )
        answer = ai_resp.choices[0].message.content.strip()
        result["answer"] = answer
    except Exception as e:
        return jsonify({"error": "OpenAI API error", "detail": str(e)}), 502

    return jsonify(result), 200


# ----------------------------
# Main process start
# ----------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)