
import hashlib
import time
import os
import json
import openai
from flask import Flask, request, jsonify, Response


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
# Main entry point
# ========================================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)