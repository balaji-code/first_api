
#!/usr/bin/env python3
"""
first_api/app.py

Clean, well-commented single-file Flask microservice that demonstrates:
 - small utility endpoints (health/echo)
 - simple OpenAI-backed endpoints (summarize, keywords, sentiment, translate)
 - a tiny RAG (retrieval) pipeline backed by a single SQLite file (docs table)

This file has been formatted and annotated for readability by a newcomer.
Keep this file in the project root next to venv/ and vector_store.db

Notes for local development:
 - Put secrets and overrides in a .env file (OPENAI_API_KEY, OPENAI_MODEL, OPENAI_EMBEDDING_MODEL,
   RAG_DB, RAG_MIN_SCORE) and use python-dotenv to load them during development.
 - Start server: `source venv/bin/activate` then `python app.py`
 - Test endpoints with curl (examples are used in your learning notes).
"""

# Standard library
import os
import json
import math
import time
import sqlite3
import re
from typing import List, Dict
import hashlib
# Third-party
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env if present (convenient for local dev)
load_dotenv()

# -----------------------------
# Basic configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DB = os.getenv("RAG_DB") or os.path.join(BASE_DIR, "vector_store.db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Minimum similarity threshold used by RAG endpoints (can be set in .env as RAG_MIN_SCORE)
try:
    MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.12"))
except Exception:
    MIN_SCORE = 0.12

print("DEBUG: RAG_DB=", RAG_DB)
print("DEBUG: OPENAI_MODEL=", OPENAI_MODEL)
print("DEBUG: EMBEDDING_MODEL=", EMBEDDING_MODEL)

# If API key is provided, create a convenience OpenAI client factory function.
if OPENAI_API_KEY:
    def make_openai_client():
        return OpenAI(api_key=OPENAI_API_KEY)
else:
    def make_openai_client():
        # return a client without api_key set (calls will then error with a clear message)
        return OpenAI()

# -----------------------------
# Flask app + JSON settings
# -----------------------------
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False  # ensures non-ascii (e.g., Telugu) is returned readable

# -----------------------------
# Small in-memory cache (simple TTL cache used by a few endpoints)
# -----------------------------
_CACHE: Dict[str, tuple] = {}
_CACHE_TTL = 60 * 60  # 1 hour


def cache_get(key: str):
    entry = _CACHE.get(key)
    if not entry:
        return None
    ts, val = entry
    if time.time() - ts > _CACHE_TTL:
        del _CACHE[key]
        return None
    return val


def cache_set(key: str, value):
    _CACHE[key] = (time.time(), value)

# -----------------------------
# Small math / embedding helpers
# -----------------------------

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


# -----------------------------
# Embeddings helper (modern OpenAI client usage)
# -----------------------------

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Return list-of-list embeddings for `texts` using the configured embedding model.
    Raises exceptions with OpenAI client errors so callers can return proper 5xx responses.
    """
    client = make_openai_client()
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    # `resp.data` is an iterable of embedding objects with .embedding
    return [item.embedding for item in resp.data]


# -----------------------------
# Simple chunking (split into readable pieces)
# -----------------------------

def simple_chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
    """Split `text` on sentence boundaries and join into human-sized chunks.
    This keeps chunks readable for the LLM and for embedding creation.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for s in sentences:
        if current_len + len(s) + 1 <= max_chars:
            current.append(s)
            current_len += len(s) + 1
        else:
            if current:
                chunks.append(' '.join(current).strip())
            # create a new chunk and keep a small overlap from previous chunk for continuity
            overlap_text = ''
            if chunks and overlap > 0:
                prev = chunks[-1]
                overlap_text = prev[-overlap:] if len(prev) > overlap else prev
            current = [overlap_text, s] if overlap_text else [s]
            current_len = sum(len(x) for x in current) + len(current) - 1
    if current:
        chunks.append(' '.join(current).strip())
    return [c for c in chunks if c]


# -----------------------------
# SQLite helpers (single 'docs' table used by RAG)
# -----------------------------

def ensure_db():
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


def load_all_embeddings() -> List[Dict]:
    """Return all rows in the docs table with parsed embeddings and metadata."""
    conn = sqlite3.connect(RAG_DB)
    cur = conn.cursor()
    cur.execute("SELECT id, text, embedding, metadata FROM docs ORDER BY id ASC")
    rows = cur.fetchall()
    conn.close()

    out = []
    for r in rows:
        _id, text, emb_json, md_json = r
        try:
            emb = json.loads(emb_json) if emb_json else None
        except Exception:
            emb = None
        try:
            md = json.loads(md_json) if md_json else {}
        except Exception:
            md = {"_raw_metadata": md_json}
        out.append({"id": _id, "text": text, "embedding": emb, "metadata": md})
    return out


# Make sure DB exists when module is imported/run
ensure_db()

# -----------------------------
# Small utility endpoints
# -----------------------------

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({"message": "Hello from your API", "status": "ok"}), 200


@app.route('/echo', methods=['POST'])
def echo():
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    data = request.get_json()
    if 'name' not in data:
        return jsonify({"error": "'name' field is required"}), 400
    if not isinstance(data['name'], str):
        return jsonify({"error": "'name' must be a string"}), 400
    return jsonify({"received": data, "message": "Valid input received"}), 200


# -----------------------------
# OpenAI-powered helper pattern
# -----------------------------

def call_chat_with_system(system_prompt: str, user_prompt: str, max_tokens: int = 256, temperature: float = 0.2):
    """Convenience wrapper that calls the modern OpenAI client and returns raw string output.

    Raises any underlying exception so endpoints can convert it to a 5xx JSON response.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not configured")
    client = make_openai_client()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        n=1,
        timeout=30,
    )
    # Robust extraction depending on returned shape
    try:
        return resp.choices[0].message.content.strip()
    except Exception:
        # older style accessibility
        return resp["choices"][0]["message"]["content"].strip()


# -----------------------------
# Example OpenAI endpoints (concise, explanatory comments)
# -----------------------------

@app.route('/summarize', methods=['POST'])
def summarize():
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({"error": "'text' required"}), 400
    try:
        system_prompt = ("You are a concise summarizer. Produce a short factual, bullet-point summary. "
                         "Do not invent facts. Keep to 3-6 bullets.")
        user_prompt = f"Summarize:\n\n{text}"
        out = call_chat_with_system(system_prompt, user_prompt, max_tokens=256, temperature=0.2)
        bullets = [l.strip() for l in out.splitlines() if l.strip()][:6]
        if not bullets:
            bullets = [s.strip() for s in re.split(r'(?<=[.!?])\s+', out) if s.strip()][:6]
        return jsonify({"summary_raw": out, "summary_bullets": bullets}), 200
    except Exception as e:
        return jsonify({"error": "OpenAI API error", "detail": str(e)}), 502


@app.route('/keywords', methods=['POST'])
def keywords():
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    text = request.get_json().get('text', '')
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "'text' required"}), 400
    key = hashlib.sha256(text.encode('utf-8')).hexdigest()
    cached = cache_get(key)
    if cached:
        return jsonify(cached), 200
    try:
        system_prompt = ("You are a strict JSON generator. Given input text, return ONLY a JSON array of up to 5 keywords.")
        user_prompt = f"Extract up to 5 keywords from the following text:\n\n{text}"
        raw = call_chat_with_system(system_prompt, user_prompt, max_tokens=150, temperature=0.0)
        m = re.search(r'(\[.*?\])', raw, flags=re.S)
        if m:
            try:
                arr = json.loads(m.group(1))
            except Exception:
                arr = [p.strip().strip('"\'') for p in re.split(r'[,\n]+', m.group(1).strip('[]')) if p.strip()]
        else:
            arr = [p.strip().strip('-• ') for p in re.split(r'[\n,]+', raw) if p.strip()][:5]
        res = {"keywords_raw": raw, "keywords": [str(x) for x in arr][:5]}
        cache_set(key, res)
        return jsonify(res), 200
    except Exception as e:
        return jsonify({"error": "Server error", "detail": str(e)}), 500


@app.route('/sentiment', methods=['POST'])
def sentiment():
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    text = request.get_json().get('text', '')
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "'text' required"}), 400
    key = hashlib.sha256(text.encode('utf-8')).hexdigest()
    cached = cache_get(key)
    if cached:
        return jsonify(cached), 200
    try:
        system_prompt = ("Return ONLY a JSON object with keys 'label' and 'score'. label ∈ {positive, neutral, negative}. score 0.0-1.0.")
        user_prompt = f"Classify the sentiment of the text and return JSON:\n\n{text}"
        raw = call_chat_with_system(system_prompt, user_prompt, max_tokens=64, temperature=0.0)
        m = re.search(r'({.*})', raw, flags=re.S)
        parsed = {}
        if m:
            try:
                parsed = json.loads(m.group(1))
            except Exception:
                pass
        # fallback simple extraction
        if 'label' not in parsed:
            lab = re.search(r'(positive|neutral|negative)', raw, flags=re.I)
            if lab:
                parsed['label'] = lab.group(1).lower()
        if 'score' not in parsed:
            sc = re.search(r'([0-9]*\.?[0-9]+)', raw)
            if sc:
                parsed['score'] = float(sc.group(1))
        label = parsed.get('label', 'neutral') if parsed.get('label') in ('positive', 'neutral', 'negative') else 'neutral'
        score = float(parsed.get('score', 0.5)) if parsed.get('score') is not None else 0.5
        res = {"sentiment_raw": raw, "label": label, "score": round(max(0.0, min(1.0, score)), 3)}
        cache_set(key, res)
        return jsonify(res), 200
    except Exception as e:
        return jsonify({"error": "Server error", "detail": str(e)}), 500


# -----------------------------
# RAG endpoints (index, upsert, bulk, docs, delete, search, query)
# -----------------------------

@app.route('/rag-index', methods=['POST'])
def rag_index():
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    payload = request.get_json()
    text = payload.get('text', '')
    if not text:
        return jsonify({"error": "'text' required"}), 400
    parent = payload.get('id')
    metadata = payload.get('metadata', {}) or {}
    chunks = simple_chunk_text(text, max_chars=int(payload.get('max_chars', 800)), overlap=int(payload.get('overlap', 100)))
    try:
        embeddings = embed_texts(chunks)
    except Exception as e:
        return jsonify({"error": "Embedding error", "detail": str(e)}), 502
    for i, (c, emb) in enumerate(zip(chunks, embeddings)):
        md = dict(metadata)
        if parent:
            md['_parent'] = parent
        md['_chunk_index'] = i
        save_doc(c, emb, md)
    return jsonify({"status": "ok", "chunks_indexed": len(chunks)}), 200


def upsert_document_to_db(doc_id: str, text: str, metadata: Dict, max_chars: int = 800, overlap: int = 100):
    """Helper used by bulk upsert: deletes previous chunks with _parent==doc_id,
    then chunk+embed+insert new ones. Returns (True, num_chunks) on success or (False, reason).
    """
    if not doc_id or not text:
        return False, "id and text required"
    # remove previous chunks for this document id
    conn = sqlite3.connect(RAG_DB)
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM docs WHERE metadata LIKE ?", (f'%"_parent":"{doc_id}"%',))
        conn.commit()
    except Exception as e:
        conn.close()
        return False, f"DB delete error: {e}"

    chunks = simple_chunk_text(text, max_chars=max_chars, overlap=overlap)
    if not chunks:
        conn.close()
        return False, "No chunks created from text"
    try:
        embeddings = embed_texts(chunks)
    except Exception as e:
        conn.close()
        return False, f"Embedding error: {e}"
    try:
        for idx, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
            md = dict(metadata or {})
            md['_parent'] = doc_id
            md['_chunk_index'] = idx
            cur.execute("INSERT INTO docs (text, embedding, metadata) VALUES (?, ?, ?)", (chunk_text, json.dumps(emb), json.dumps(md)))
        conn.commit()
        conn.close()
        return True, len(chunks)
    except Exception as e:
        conn.close()
        return False, f"DB insert error: {e}"


@app.route('/rag-upsert', methods=['POST'])
def rag_upsert():
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    payload = request.get_json()
    doc_id = (payload.get('id') or '').strip()
    text = (payload.get('text') or '').strip()
    metadata = payload.get('metadata', {}) or {}
    if not doc_id:
        return jsonify({"error": "'id' required"}), 400
    if not text:
        return jsonify({"error": "'text' required"}), 400
    ok, info = upsert_document_to_db(doc_id, text, metadata)
    if not ok:
        return jsonify({"error": "upsert failed", "detail": info}), 500
    return jsonify({"status": "ok", "id": doc_id, "chunks_indexed": info}), 200


@app.route('/rag-bulk-index', methods=['POST'])
def rag_bulk_index():
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    payload = request.get_json()
    docs = payload.get('docs')
    if not isinstance(docs, list) or not docs:
        return jsonify({"error": "'docs' must be a non-empty list"}), 400
    summary = {"processed": 0, "succeeded": 0, "failed": 0, "details": [], "total_chunks_indexed": 0}
    for d in docs:
        doc_id = (d.get('id') or '').strip()
        text = (d.get('text') or '').strip()
        metadata = d.get('metadata', {}) or {}
        summary['processed'] += 1
        ok, info = upsert_document_to_db(doc_id, text, metadata)
        if ok:
            summary['succeeded'] += 1
            summary['details'].append({"id": doc_id, "status": "ok", "chunks_indexed": info})
            summary['total_chunks_indexed'] += info
        else:
            summary['failed'] += 1
            summary['details'].append({"id": doc_id, "status": "failed", "reason": info})
    return jsonify(summary), 200


@app.route('/rag-docs', methods=['GET'])
def rag_docs():
    include_embeddings = request.args.get('include_embeddings', 'false').lower() in ('1', 'true', 'yes')
    rows = load_all_embeddings()
    docs = []
    for r in rows:
        emb_len = len(r['embedding']) if r.get('embedding') else 0
        docs.append({"id": r['id'], "text": r['text'], "embedding_length": emb_len, "embedding": (r['embedding'] if include_embeddings else None), "metadata": r.get('metadata', {})})
    return jsonify({"count": len(docs), "chunks": docs}), 200


@app.route('/rag-delete', methods=['POST'])
def rag_delete():
    data = request.get_json() or {}
    if 'id' not in data:
        return jsonify({"error": "Provide 'id' to delete"}), 400
    chunk_id = data['id']
    conn = sqlite3.connect(RAG_DB)
    cur = conn.cursor()
    cur.execute('SELECT id FROM docs WHERE id = ?', (chunk_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "No such chunk ID"}), 404
    cur.execute('DELETE FROM docs WHERE id = ?', (chunk_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "deleted", "id": chunk_id}), 200


@app.route('/rag-search', methods=['POST'])
def rag_search():
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    payload = request.get_json()
    query = (payload.get('query') or '').strip()
    if not query:
        return jsonify({"error": "'query' required"}), 400
    top_k = max(1, min(int(payload.get('top_k', 3)), 20))
    metadata_filter = payload.get('metadata', {}) or {}
    try:
        q_emb = embed_texts([query])[0]
    except Exception as e:
        return jsonify({"error": "Embedding error", "detail": str(e)}), 502
    rows = load_all_embeddings()
    scored = []
    for r in rows:
        emb = r.get('embedding')
        if not emb:
            continue
        md = r.get('metadata', {}) or {}
        # metadata filter: each key must match exactly
        ok = True
        if isinstance(metadata_filter, dict):
            for k, v in metadata_filter.items():
                if md.get(k) != v:
                    ok = False
                    break
        if not ok:
            continue
        score = cosine_similarity(q_emb, emb)
        scored.append({"id": r['id'], "text": r['text'], "metadata": md, "score": float(score)})
    scored.sort(key=lambda x: x['score'], reverse=True)
    top_matches = [s for s in scored if s['score'] >= MIN_SCORE][:top_k]
    return jsonify({"query": query, "top_matches": top_matches, "count_scored": len(scored)}), 200


@app.route('/rag-query', methods=['POST'])
def rag_query():
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    payload = request.get_json()
    query = (payload.get('query') or '').strip()
    if not query:
        return jsonify({"error": "'query' required"}), 400
    top_k = max(1, min(int(payload.get('top_k', 3)), 10))
    use_llm = bool(payload.get('use_llm', True))
    try:
        q_emb = embed_texts([query])[0]
    except Exception as e:
        return jsonify({"error": "Embedding error", "detail": str(e)}), 502
    rows = load_all_embeddings()
    scored = []
    for r in rows:
        emb = r.get('embedding')
        if not emb:
            continue
        score = cosine_similarity(q_emb, emb)
        scored.append({"id": r['id'], "text": r['text'], "metadata": r.get('metadata', {}), "score": float(score)})
    scored.sort(key=lambda x: x['score'], reverse=True)
    top = [s for s in scored if s['score'] >= MIN_SCORE][:top_k]
    result = {"query": query, "top_matches": top}
    if not top:
        result['answer'] = "I don't know (no relevant documents match)."
        return jsonify(result), 200
    if not use_llm:
        return jsonify(result), 200
    # assemble context for LLM
    MAX_CONTEXT_CHARS = 4000
    parts = []
    total = 0
    for t in top:
        txt = t['text']
        if total + len(txt) > MAX_CONTEXT_CHARS:
            remain = MAX_CONTEXT_CHARS - total
            if remain <= 0:
                break
            txt = txt[:remain]
        parts.append(txt)
        total += len(txt)
    context = '\n\n---\n\n'.join(parts)
    system_prompt = ("You are a helpful assistant. Use ONLY the provided context to answer. "
                     "If the answer is not in the context, say you don't know. Be concise.")
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"
    try:
        client = make_openai_client()
        ai_resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_tokens=300,
            temperature=0.0,
            timeout=30,
        )
        answer = ai_resp.choices[0].message.content.strip()
        result['answer'] = answer
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": "OpenAI API error", "detail": str(e)}), 502


# -----------------------------
# Run the server
# -----------------------------
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5050, debug=True)