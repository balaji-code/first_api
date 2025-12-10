
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
import requests
from openai import OpenAI
import faiss
import numpy as np
import pickle
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
RERANK_MODEL = os.getenv("RERANK_MODEL", OPENAI_MODEL)
CHUNKER_MODE = os.getenv("RAG_CHUNKER", "semantic").lower()
try:
    CHUNK_SIM_THRESHOLD = float(os.getenv("CHUNK_SIM_THRESHOLD", "0.55"))
except Exception:
    CHUNK_SIM_THRESHOLD = 0.55
# Minimum similarity threshold used by RAG endpoints (can be set in .env as RAG_MIN_SCORE)
try:
    MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.12"))
except Exception:
    MIN_SCORE = 0.12
# RAG_CONFIDENCE_THRESHOLD: minimum conservative score (0..1) required to let the LLM answer; lower values make the system more permissive
RAG_CONFIDENCE_THRESHOLD = float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.6"))

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
# ----------------- Light-weight Context Compression -----------------
import re
from functools import lru_cache

# Simple in-memory cache for compressed chunks (id -> summary)
# For production you'd persist this (SQLite table / KV store).
COMPRESS_CACHE = {}

# Basic sentence tokenizer (keeps punctuation). Splits on .!? + whitespace.
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+')

def simple_compress_text(text: str, max_sentences: int = 2) -> str:
    """
    Tiny deterministic compressor: returns up-to max_sentences sentences
    from the start of `text`. Reason: simple, fast, and predictable for learning.
    """
    if not text:
        return ""
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    # If text lacks punctuation, fall back to character truncation
    if len(parts) == 1 and len(parts[0]) > 400:
        return parts[0][:400].rstrip() + "..."
    selected = parts[:max_sentences]
    return " ".join(s.strip() for s in selected if s.strip())

def get_compressed_chunk(chunk_id: int, text: str, max_sentences: int = 2) -> str:
    """
    Use COMPRESS_CACHE keyed by chunk id (int) to avoid recomputing.
    If chunk_id is None or not int, fall back to compress without caching.
    """
    try:
        key = int(chunk_id)
    except Exception:
        return simple_compress_text(text, max_sentences=max_sentences)

    if key in COMPRESS_CACHE:
        return COMPRESS_CACHE[key]

    summary = simple_compress_text(text, max_sentences=max_sentences)
    COMPRESS_CACHE[key] = summary
    return summary

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

def semantic_chunk_text(text: str, max_chars: int = 1200, overlap: int = 100,
                        sim_threshold: float = 0.55) -> List[str]:
    """
    Split text into sentences, then merge adjacent sentences when semantically similar.
    Added debug logging to show per-sentence similarities and merge decisions.
    - max_chars: cap chunk length (if merged chunk grows beyond this, force split)
    - overlap: unused here (kept for API parity)
    - sim_threshold: cosine similarity threshold for merging (0-1)

    Returns list[str] chunks.
    """
    # 1) Sentence-split (same as before)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    if not sentences:
        return []

    # 2) Embed all sentences in one batch
    try:
        sent_embs = embed_texts(sentences)  # [[...], [...], ...]
    except Exception as e:
        # fallback to returning simple sentence chunks if embedding fails
        print("DEBUG: semantic_chunk_text embedding failed:", e)
        return sentences

    # 3) Build chunks by merging adjacent sentences with similarity to current centroid
    chunks = []
    current_sentences = [sentences[0]]
    current_vecs = [np.array(sent_embs[0], dtype="float32")]
    current_len = len(current_sentences[0])

    # debug header
    print(f"DEBUG: semantic_chunk_text: {len(sentences)} sentences, sim_threshold={sim_threshold}")

    for i in range(1, len(sentences)):
        s = sentences[i]
        v = np.array(sent_embs[i], dtype="float32")

        # centroid of current chunk
        centroid = np.mean(np.stack(current_vecs, axis=0), axis=0)
        # normalize for cosine
        cn = np.linalg.norm(centroid)
        vn = np.linalg.norm(v)
        sim = 0.0
        if cn > 0 and vn > 0:
            sim = float(np.dot(centroid, v) / (cn * vn))

        # decide whether to merge
        will_merge = (sim >= sim_threshold) and (current_len + len(s) + 1 <= max_chars)
        print(f"DEBUG: sent[{i}] preview='{s[:60]}', sim={sim:.4f}, current_len={current_len}, will_merge={will_merge}")

        if will_merge:
            current_sentences.append(s)
            current_vecs.append(v)
            current_len += len(s) + 1
        else:
            # finalize current chunk
            finalized = " ".join(current_sentences).strip()
            print(f"DEBUG: finalizing chunk (len={len(finalized)}) preview='{finalized[:80]}'")
            chunks.append(finalized)
            # start new chunk
            current_sentences = [s]
            current_vecs = [v]
            current_len = len(s)

    # append final chunk
    if current_sentences:
        final = " ".join(current_sentences).strip()
        print(f"DEBUG: final chunk (len={len(final)}) preview='{final[:80]}'")
        chunks.append(final)

    return chunks

def choose_chunker(text, max_chars=800, overlap=100):
    """
    Wrapper that chooses the chunker based on CHUNKER_MODE.
    - 'semantic'  → semantic_chunk_text using CHUNK_SIM_THRESHOLD
    - 'simple'    → simple_chunk_text
    """
    if CHUNKER_MODE == "semantic":
        return semantic_chunk_text(text, max_chars=max_chars, overlap=overlap, sim_threshold=CHUNK_SIM_THRESHOLD)
    return simple_chunk_text(text, max_chars=max_chars, overlap=overlap)
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

# ---------- FAISS index helpers ----------
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "vector_store.faiss")
FAISS_IDMAP_PATH = os.path.join(BASE_DIR, "faiss_idmap.pickle")
VECTOR_DIM = 1536  # must match embedding model

# In-memory structures
_faiss_index = None
_id_to_rowid = {}   # map faiss internal id -> DB row id (int)

def init_faiss_index():
    global _faiss_index, _id_to_rowid
    # If persisted index exists, try to load it
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_IDMAP_PATH):
        try:
            _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            with open(FAISS_IDMAP_PATH, "rb") as f:
                _id_to_rowid = pickle.load(f)
            print("DEBUG: Loaded FAISS index from disk, entries:", len(_id_to_rowid))
            return
        except Exception as e:
            print("DEBUG: Failed to load persisted FAISS index:", e)

    # otherwise create a fresh Flat index (simple)
    _faiss_index = faiss.IndexFlatIP(VECTOR_DIM)  # use inner product for cosine if vectors are normalized
    _id_to_rowid = {}
    rebuild_faiss_index()  # populate from DB

def rebuild_faiss_index():
    """Read embeddings from DB and (re)build FAISS and id map."""
    global _faiss_index, _id_to_rowid
    rows = load_all_embeddings()
    vecs = []
    ids = []
    for r in rows:
        emb = r.get("embedding")
        if not emb:
            continue
        # convert to float32 numpy
        arr = np.array(emb, dtype="float32")
        # If using IP similarity for cosine, normalize first:
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        vecs.append(arr)
        ids.append(r["id"])
    if vecs:
        xb = np.vstack(vecs)
        # recreate index and add data
        _faiss_index = faiss.IndexFlatIP(VECTOR_DIM)
        _faiss_index.add(xb)
        # create id map mapping internal index position -> DB row id
        _id_to_rowid = {i: ids[i] for i in range(len(ids))}
    else:
        _faiss_index = faiss.IndexFlatIP(VECTOR_DIM)
        _id_to_rowid = {}
    persist_faiss_index()

def persist_faiss_index():
    """Save index + id map to disk."""
    try:
        faiss.write_index(_faiss_index, FAISS_INDEX_PATH)
        with open(FAISS_IDMAP_PATH, "wb") as f:
            pickle.dump(_id_to_rowid, f)
    except Exception as e:
        print("DEBUG: persist_faiss_index error:", e)

def add_vector_to_faiss(db_row_id: int, emb: List[float]):
    """Append a single vector to FAISS and map its index -> db_row_id."""
    global _faiss_index, _id_to_rowid
    arr = np.array(emb, dtype="float32")
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    arr = arr.reshape(1, -1)
    # FAISS IndexFlatIP does not support ids; we append and map by position
    _faiss_index.add(arr)
    new_pos = int(_faiss_index.ntotal) - 1
    _id_to_rowid[new_pos] = db_row_id
    persist_faiss_index()

def search_faiss_by_vector(query_emb: List[float], top_k: int = 5):
    """Return list of tuples (db_row_id, score)."""
    global _faiss_index, _id_to_rowid
    if _faiss_index is None or _faiss_index.ntotal == 0:
        return []
    q = np.array(query_emb, dtype="float32")
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    q = q.reshape(1, -1)
    D, I = _faiss_index.search(q, top_k)  # D: distances/scores, I: positions
    out = []
    for score, pos in zip(D[0], I[0]):
        if pos < 0:
            continue
        db_id = _id_to_rowid.get(int(pos))
        out.append((db_id, float(score)))
    return out
# Make sure DB exists when module is imported/run

ensure_db()
init_faiss_index()
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
# Reranker helpers (LLM-based)
# -----------------------------
import math


def extract_first_json_array(text: str) -> str | None:
    """Return the first JSON array substring (including brackets) found in text.
    This does a simple bracket-depth scan to avoid greedy regex matches.
    Returns None if no complete array is found.
    """
    start = text.find('[')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None


def lexical_overlap_score(query: str, passage: str) -> float:
    """Simple normalized token overlap between query and passage (0..1).
    This is a cheap signal to penalize candidates that lack lexical overlap.
    """
    q_words = set(re.findall(r"\w+", query.lower()))
    p_words = set(re.findall(r"\w+", passage.lower()))
    if not q_words or not p_words:
        return 0.0
    inter = q_words & p_words
    return float(len(inter)) / float(max(1, len(q_words)))


# --- Entity in text helper ---
def entity_in_text(entity: str, txt: str) -> bool:
    """Return True if the entity (or its simple plural) appears as a whole word in txt.
    Accepts a pipe-separated entity string (e.g. 'apple|apples|fruit|fruits').
    """
    if not entity or not txt:
        return False
    txt_l = txt.lower()
    # allow multiple tokens separated by | in entity_hint
    for token in str(entity).split('|'):
        t = token.strip().lower()
        if not t:
            continue
        # exact word match
        if re.search(rf"\b{re.escape(t)}\b", txt_l):
            return True
        # plural heuristic: add 's' if not present
        if not t.endswith('s') and re.search(rf"\b{re.escape(t)}s\b", txt_l):
            return True
    return False


# --- Entity hint expansion helper ---
def expand_entity_hint(entity_hint: str) -> str:
    """
    Expand a short entity hint returned by the rewrite step into a pipe-separated
    list of tokens we will try to match in documents. This is a lightweight,
    deterministic approach that covers common pluralization and a handful of
    useful domain mappings (e.g. 'fruit' -> 'apple|apples|fruit|fruits').
    Returns a string like 'apple|apples|fruit|fruits' suitable for passing into
    entity_in_text.
    """
    if not entity_hint:
        return ''
    e = entity_hint.strip().lower()
    # simple domain-specific expansions (small deterministic map)
    mapping = {
        'fruit': 'apple|apples|fruit|fruits',
        'dog': 'dog|dogs|puppy|puppies',
        'car': 'car|cars|vehicle|vehicles',
        'computer': 'computer|computers|pc|pcs',
    }
    if e in mapping:
        return mapping[e]
    # otherwise, produce variants: original, singular/plural heuristic
    parts = [e]
    if not e.endswith('s'):
        parts.append(e + 's')
    else:
        parts.append(e[:-1])
    # where helpful, include the plain stem (remove common suffixes)
    stem = re.sub(r'(ing|ed|es)$', '', e)
    if stem and stem not in parts:
        parts.append(stem)
    # dedupe and join
    seen = []
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            seen.append(p)
    return '|'.join(seen)


# --- Simple inversion penalty heuristic ---
def simple_inversion_penalty(query: str, passage: str) -> float:
    """
    Conservatively detect simple subject-object inversions around the verb 'grow'
    and apply a heavy penalty (0.05) so inverted passages are downweighted.

    Improvements made in this replacement:
    - Added detection of explicit contrast words (e.g., "not", "instead") to slightly
      relax the penalty when the passage explicitly negates or contrasts a claim (returns 0.2)
      — this helps avoid false positives on sentences like "They grow on plants, not trees." which
      are contrastive rather than inverted facts.
    - Keep strong penalty (0.05) when a real inversion is detected.
    - Be permissive (1.0) when we cannot parse a clear subj/obj or when inputs are empty.
    """
    try:
        if not query or not passage:
            return 1.0
        q_low = query.lower()
        p_low = passage.lower()

        # look for simple subject / object patterns around 'grow'
        m_subj = re.search(r"\b([a-z0-9_'-]{1,40})\b\s+grow(?:s|ing)?\b", p_low)
        m_obj = re.search(r"grow(?:s|ing)?\b\s+on\s+\b([a-z0-9_'-]{1,40})\b", p_low)

        if not (m_subj and m_obj):
            return 1.0

        subj = m_subj.group(1)
        obj = m_obj.group(1)

        # Tokenize query safely
        q_tokens = set(re.findall(r"\w+", q_low))
        subj_in_q = subj in q_tokens
        obj_in_q = obj in q_tokens

        # If neither appears in the query, be permissive
        if not subj_in_q and not obj_in_q:
            return 1.0

        # Check for explicit contrast words in the passage which often indicate a correction
        contrast_words = (" not ", " instead ", " rather ", " but ")
        has_contrast = any(w in p_low for w in contrast_words)

        # If only the object appears and it is likely a role-swap (inversion) -> heavy penalty
        if obj_in_q and not subj_in_q:
            # slightly relax penalty if passage explicitly contrasts (e.g. "not trees")
            return 0.2 if has_contrast else 0.05

        # If both appear in the query, this is an ambivalent/leading question.
        # Apply strong penalty but relax slightly when passage contains explicit contrast words.
        if subj_in_q and obj_in_q:
            return 0.2 if has_contrast else 0.05

        # Other cases (only subj in query) -> no penalty
        return 1.0

    except Exception:
        return 1.0


def rerank_candidates_with_llm(
    query: str, candidates: List[Dict], model: str = None, max_tokens: int = 512
) -> List[float]:
    """
    Ask the LLM to score each candidate for relevance to the query.
    - candidates: list of {"id": <int>, "text": <str>, "metadata": {...}, "score": <float>}
    - returns: list of floats (same order as candidates) between 0.0 and 1.0

    This version uses a stricter system prompt with a short example and a robust
    JSON-array extraction helper as a fallback to avoid brittle regex captures.
    """
    if not candidates:
        return []

    model = model or OPENAI_MODEL
    # Build a compact prompt that enumerates candidates.
    truncated_cands = []
    for i, c in enumerate(candidates):
        text = c.get("text", "")
        if len(text) > 600:
            text = text[:600] + " …"
        truncated_cands.append((i, text))

    system_prompt = (
        "You are a strict relevance scorer. Given a user's query and a small list of candidate passages, "
        "return ONLY a JSON array of numbers between 0.0 and 1.0 (inclusive) representing the relevance of each "
        "candidate to the query. The array must have the same length and order as the candidates. "
        "Use at most three decimal places. Do NOT add any commentary, explanation, or trailing text.\n\n"
        "EXAMPLE:\n"
        "QUERY: Where do apples grow?\n"
        "CANDIDATES:\n"
        "0. Apples are fruits. They grow on trees.\n"
        "1. Trees grow on apples.\n\n"
        "RESPONSE:\n"
        "[1.000, 0.000]\n"
    )

    parts = [f"QUERY: {query}", "", "CANDIDATES:"]
    for idx, text in truncated_cands:
        parts.append(f"{idx}. {text}")
    user_prompt = "\n".join(parts)

    try:
        client = make_openai_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max(40),
            temperature=0.0,
            n=1,
            timeout=30,
        )
        raw = ""
        try:
            raw = resp.choices[0].message.content.strip()
        except Exception:
            raw = resp["choices"][0]["message"]["content"].strip()

        # Debug log (truncated) for troubleshooting in development
        print("DEBUG: reranker raw (truncated 800):", raw[:800])

        # Extract first JSON array robustly
        arr_text = extract_first_json_array(raw)
        if not arr_text:
            raise ValueError("Reranker did not return JSON array.")
        scores = json.loads(arr_text)

        # validate and normalize
        out = []
        for s in scores:
            try:
                f = float(s)
            except Exception:
                f = 0.0
            f = max(0.0, min(1.0, f))
            out.append(round(f, 6))

        if len(out) != len(candidates):
            raise ValueError("Reranker length mismatch.")
        return out

    except Exception as e:
        # fallback: compute normalized cosine similarity between query and candidate embeddings (if available)
        print("DEBUG: reranker LLM failed, falling back to cosine. Error:", e)
        fallback = []
        try:
            q_emb = embed_texts([query])[0]
        except Exception:
            q_emb = None
        for c in candidates:
            emb = c.get("embedding")
            if emb and q_emb:
                try:
                    fallback.append(cosine_similarity(q_emb, emb))
                except Exception:
                    fallback.append(float(c.get("score", 0.0)))
            else:
                fallback.append(float(c.get("score", 0.0)))
        maxv = max(fallback) if fallback else 1.0
        if maxv <= 0:
            maxv = 1.0
        normalized = [min(1.0, max(0.0, f / maxv)) for f in fallback]
        return normalized



def rerank_and_sort_candidates(
    query: str, rows: List[Dict], top_k: int = 5, use_reranker: bool = True
):
    """
    rows: list of dicts with keys {id, text, metadata, score} where 'score' is FAISS score or similar.
    Returns top_k rows sorted by reranker score if use_reranker True, otherwise by score.
    Adds 'rerank_score' (and 'combined_score') to each returned row when reranker used.
    """
    if not rows:
        return []

    # keep a wider candidate set for reranking
    candidates = rows[: top_k * 3]

    if use_reranker:
        # call the LLM reranker to score candidates
        rerank_scores = rerank_candidates_with_llm(query, candidates)

        # compute lexical overlap + apply inversion penalty + combine scores
        alpha = 0.65  # weight for reranker score (reduced to avoid over-trusting LLM reranker)
        entity_bonus = 0.06  # smaller entity boost
        for c, s in zip(candidates, rerank_scores):
            # base reranker score (0..1)
            rscore = float(s)
            # apply simple inversion penalty (cheap heuristic)
            penalty = simple_inversion_penalty(query, c.get('text', ''))
            rscore = rscore * penalty
            c['rerank_score'] = round(float(rscore), 6)
            # lexical overlap signal
            lex = lexical_overlap_score(query, c.get('text', ''))
            c['lexical_overlap'] = round(float(lex), 6)
            ent = int(c.get('entity_match', 0))
            # small conditional entity signal applied only when lexical overlap is meaningful
            entity_signal = entity_bonus if (ent and lex >= 0.25) else 0.0
            # include a small contribution from the original semantic (FAISS) score as a tie-breaker
            orig_score = float(c.get('score', 0.0))
            c['combined_score'] = float(
                alpha * c['rerank_score']
                + (1 - alpha) * c['lexical_overlap']
                + entity_signal
                + 0.15 * orig_score
            )
            # helpful debug logging when combined_score is high but lexical overlap is very low
            if c['combined_score'] > 0.8 and c['lexical_overlap'] < 0.15:
                app.logger.warning("High combined_score but low lexical overlap for candidate id=%s", c.get('id'))

        # sort by combined_score, then lexical_overlap, then original FAISS score
        candidates.sort(key=lambda x: (x.get('combined_score', 0.0), x.get('lexical_overlap', 0.0), x.get('score', 0.0)), reverse=True)
    else:
        candidates.sort(key=lambda x: x.get('score', 0.0), reverse=True)

    return candidates[:top_k]
# -----------------------------
# Evidence scoring helper
# -----------------------------

# Get minimum required evidence score from environment variable, default to 0.0 if not set or invalid.
try:
    EVIDENCE_MIN = float(os.getenv("RAG_EVIDENCE_MIN", "0.0"))
except Exception:
    EVIDENCE_MIN = 0.0

def compute_evidence_score(
    query: str,
    candidate: dict,
    q_emb: list | None = None,
    weights: dict | None = None
) -> dict:
    """
    Compute evidence components and a combined evidence score for a candidate.

    Returns:
        dict: {S, L, M, C, evidence_score}
            S: semantic similarity (float 0..1)
            L: lexical overlap (float 0..1)
            M: entity match (1 or 0)
            C: contradiction/inversion penalty flag (1 or 0)
            evidence_score: weighted combination (float)

    Args:
        query (str): original (or rewritten) query
        candidate (dict): has keys 'text', 'metadata', 'embedding', 'score'
        q_emb (list or None): precomputed query embedding (recommended)
        weights (dict or None): keys alpha, beta, gamma, delta

    Note: Keep this deterministic and conservative during learning.
    """

    # Set default weights if not provided
    if weights is None:
        weights = {"alpha": 0.55, "beta": 0.25, "gamma": 0.10, "delta": 0.80}

    text = (candidate.get("text") or "").strip()
    md = candidate.get("metadata") or {}

    # 1. Semantic similarity S (range 0..1)
    S = 0.0
    try:
        if q_emb is None:
            # Compute embedding for the query if not provided
            q_emb = embed_texts([query])[0]
        c_emb = candidate.get("embedding")
        if c_emb:
            S = float(cosine_similarity(q_emb, c_emb))
        else:
            # Fallback: use FAISS score as proxy, clamp to [0,1]
            S = max(0.0, min(1.0, float(candidate.get("score", 0.0))))
    except Exception:
        # If embedding fails, fallback to FAISS score
        S = max(0.0, min(1.0, float(candidate.get("score", 0.0))))

    # 2. Lexical overlap L (fraction of query tokens covered, 0..1)
    q_words = set(re.findall(r"\w+", query.lower()))
    t_words = set(re.findall(r"\w+", text.lower()))
    L = float(len(q_words & t_words)) / max(1, len(q_words))

    # 3. Entity match M (1 if candidate matched query entity, 0 otherwise)
    M = 1.0 if int(candidate.get("entity_match", 0)) else 0.0

    # 4. Contradiction/inversion penalty flag C_flag (1 = problematic, else 0)
    C_flag = 0
    try:
        penalty = simple_inversion_penalty(query, text)
        # If inversion penalty is very low, mark as contradiction
        if penalty < 0.2:
            C_flag = 1
        # Add structured debug logging so we can trace why candidates are penalized.
        try:
            cid = candidate.get('id') if isinstance(candidate, dict) else None
            app.logger.debug(
                "compute_evidence_score: candidate_id=%s penalty=%s C=%s query_preview=%s passage_preview=%s",
                str(cid),
                str(round(penalty, 4)),
                str(C_flag),
                (query[:120] + '...') if len(query) > 120 else query,
                (text[:120] + '...') if len(text) > 120 else text,
            )
        except Exception:
            pass
    except Exception:
        # On error, do not penalize
        C_flag = 0

    # 5. Compute weighted sum for combined evidence_score
    alpha = weights["alpha"]
    beta = weights["beta"]
    gamma = weights["gamma"]
    delta = weights["delta"]

    evidence_score = alpha * S + beta * L + gamma * M - delta * float(C_flag)

    # 6. Clamp component values for numeric stability and return as rounded values
    return {
        "S": round(float(S), 6),
        "L": round(float(L), 6),
        "M": int(M),
        "C": int(C_flag),
        "evidence_score": round(float(evidence_score), 6),
    }

# -----------------------------
# Example OpenAI endpoints (concise, explanatory comments)
# -----------------------------

def rerank_candidates(query: str, candidates: List[Dict]) -> List[Dict]:
    """
    Backwards-compatible wrapper expected by `rag_query`.
    Tries to use the LLM-based reranker pipeline (rerank_and_sort_candidates). If
    anything goes wrong, it falls back to a safe deterministic ordering by `score`.

    Returns a list of candidate dicts (same shape as input) with an added
    'rerank_score' key when available and sorted by relevance descending.
    """
    if not candidates:
        return []
    try:
        # Use the existing reranker + sorting helper to produce a ranked list.
        ranked = rerank_and_sort_candidates(query, candidates, top_k=len(candidates), use_reranker=True)
        # Ensure each returned candidate has a numeric rerank_score
        for c in ranked:
            if 'rerank_score' not in c:
                c['rerank_score'] = float(c.get('score', 0.0))
        return ranked
    except Exception as e:
        # Defensive fallback: log and return by original FAISS score
        app.logger.exception("rerank_candidates wrapper failed, falling back to score sort: %s", e)
        out = sorted(candidates, key=lambda x: x.get('score', 0.0), reverse=True)
        for c in out:
            if 'rerank_score' not in c:
                try:
                    c['rerank_score'] = float(c.get('score', 0.0))
                except Exception:
                    c['rerank_score'] = 0.0
        return out

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
    chunks = choose_chunker(text, max_chars=int(payload.get('max_chars', 800)),
                        overlap=int(payload.get('overlap', 100)))
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

    chunks = choose_chunker(text, max_chars=max_chars, overlap=overlap)
    if not chunks:
        conn.close()
        return False, "No chunks created from text"
    try:
        embeddings = embed_texts(chunks)
    except Exception as e:
        conn.close()
        return False, f"Embedding error: {e}"
    inserted_row_ids = []
    try:
        for idx, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
            md = dict(metadata or {})
            md['_parent'] = doc_id
            md['_chunk_index'] = idx
            cur.execute(
                "INSERT INTO docs (text, embedding, metadata) VALUES (?, ?, ?)",
                (chunk_text, json.dumps(emb), json.dumps(md))
            )
            # get DB row id of the inserted chunk
            row_id = cur.lastrowid
            inserted_row_ids.append((row_id, emb))
        # commit once for the batch
        conn.commit()
        # Now add vectors to FAISS (do this after DB commit)
        for row_id, emb in inserted_row_ids:
            try:
                add_vector_to_faiss(row_id, emb)
            except Exception as e:
                # don't fail the whole upsert on faiss add error; surface debug output
                print("DEBUG: add_vector_to_faiss failed for row", row_id, ":", e)
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

    # 1) compute query embedding (used for evidence S and optional semantic checks)
    try:
        q_emb = embed_texts([query])[0]
    except Exception as e:
        return jsonify({"error": "Embedding error", "detail": str(e)}), 502

    # 2) Use FAISS to get candidate DB row ids + scores
    try:
        faiss_hits = search_faiss_by_vector(q_emb, top_k=top_k * 3)  # fetch extra to allow filtering
    except Exception as e:
        return jsonify({"error": "FAISS search error", "detail": str(e)}), 502

    conn = sqlite3.connect(RAG_DB)
    cur = conn.cursor()
    candidates = []
    count_scored = 0
    try:
        for db_row_id, score in faiss_hits:
            count_scored += 1
            try:
                cur.execute("SELECT id, text, metadata FROM docs WHERE id = ?", (db_row_id,))
                row = cur.fetchone()
            except Exception:
                row = None
            if not row:
                continue
            _id, text, md_json = row
            try:
                md = json.loads(md_json) if md_json else {}
            except Exception:
                md = {"_raw_metadata": md_json}

            # apply metadata filter
            ok = True
            if isinstance(metadata_filter, dict):
                for k, v in metadata_filter.items():
                    if md.get(k) != v:
                        ok = False
                        break
            if not ok:
                continue

            # enforce MIN_SCORE threshold (faiss score)
            if score < MIN_SCORE:
                continue

            cand = {"id": _id, "text": text, "metadata": md, "score": float(score)}
            # compute evidence for this candidate (S,L,M,C,evidence_score)
            try:
                ev = compute_evidence_score(query, {**cand, "embedding": None}, q_emb=q_emb)
            except Exception:
                ev = {"S": 0.0, "L": 0.0, "M": 0, "C": 0, "evidence_score": 0.0}
            cand["evidence"] = ev
            cand["evidence_score"] = ev["evidence_score"]
            candidates.append(cand)
    finally:
        conn.close()

    # remove contradictory candidates (C==1)
    filtered = [c for c in candidates if c.get("evidence", {}).get("C", 0) == 0]

    # apply minimum evidence threshold if configured (EVIDENCE_MIN)
    filtered = [c for c in filtered if c.get("evidence_score", -999) >= EVIDENCE_MIN]

    # sort by evidence_score then faiss score
    filtered.sort(key=lambda x: (x.get("evidence_score", 0.0), x.get("score", 0.0)), reverse=True)

    # pick top_k
    top_matches = filtered[:top_k]

    # Optional: compress top match texts for display (safe, non-destructive)
    compress_context = bool(payload.get("compress_context", False))
    try:
        compress_max_sentences = int(payload.get("compress_max_sentences", 1))
    except Exception:
        compress_max_sentences = 1

    if compress_context:
        compressor = globals().get("get_compressed_chunk")
        for m in top_matches:
            orig_text = m.get("text", "")
            try:
                if callable(compressor):
                    summary = compressor(m.get("id"), orig_text, max_sentences=compress_max_sentences)
                else:
                    summary = simple_compress_text(orig_text, max_sentences=compress_max_sentences)
            except Exception:
                summary = orig_text

            m["original_text_length"] = len(orig_text)
            m["compressed_text_length"] = len(summary)
            m["compression_ratio"] = round(len(summary) / (len(orig_text) + 1), 3)
            m["text"] = summary

    return jsonify({"query": query, "top_matches": top_matches, "count_scored": count_scored}), 200

@app.route('/rag-query', methods=['POST'])
def rag_query():
    """
    Robust wrapper over your previous rag_query.
    Always returns a valid Flask response and logs unexpected errors for debugging.
    POST JSON:
      {"query":"...", "top_k": 3, "use_llm": true/false, "use_reranker": true/false}
    """
    try:
        if not request.is_json:
            return jsonify({"error": "JSON body required"}), 400

        payload = request.get_json()
        query = (payload.get('query') or '').strip()
        original_query = query  # keep original for output when rewrite is used
        if not query:
            return jsonify({"error": "'query' required"}), 400

        top_k = max(1, min(int(payload.get('top_k', 3)), 10))
        use_llm = bool(payload.get('use_llm', True))
        use_reranker = bool(payload.get('use_reranker', False))
        use_rewrite = bool(payload.get("use_rewrite", False))
        entity_hint = None

        if use_rewrite:
            try:
                rew_resp = requests.post(
                    "http://127.0.0.1:5050/rewrite-query",
                    json={"query": query}, timeout=10
                )
                rew_resp_json = rew_resp.json() if rew_resp.ok else {}
                if 'rewritten' in rew_resp_json:
                    query = rew_resp_json['rewritten']
                    entity_hint = rew_resp_json.get('entity')
            except Exception as e:
                app.logger.debug("Rewrite failed: %s", e)

        # preserve original for client visibility
        original_query_saved = original_query

        # 1) embed query
        try:
            q_emb = embed_texts([query])[0]
        except Exception as e:
            app.logger.exception("Embedding error in rag_query")
            return jsonify({"error": "Embedding error", "detail": str(e)}), 502

        # 2) FAISS search (get a few extra if we will filter / rerank)
        try:
            fetch_k = top_k * 2 if use_reranker else top_k
            faiss_hits = search_faiss_by_vector(q_emb, top_k=fetch_k)
        except Exception as e:
            app.logger.exception("FAISS search error in rag_query")
            return jsonify({"error": "FAISS search error", "detail": str(e)}), 502

        # 3) fetch DB rows for hits and assemble candidate list
        rows = []
        conn = sqlite3.connect(RAG_DB)
        cur = conn.cursor()
        try:
            for db_row_id, score in faiss_hits:
                # skip any invalid id/score
                if db_row_id is None:
                    continue
                try:
                    # fetch id, text, embedding, metadata
                    cur.execute(
                        "SELECT id, text, embedding, metadata FROM docs WHERE id = ?", (db_row_id,)
                    )
                    r = cur.fetchone()
                    if not r:
                        continue
                    _id, text, emb_json, md_json = r
                    # parse embedding JSON if present
                    try:
                        emb = json.loads(emb_json) if emb_json else None
                    except Exception:
                        emb = None
                    try:
                        md = json.loads(md_json) if md_json else {}
                    except Exception:
                        md = {"_raw_metadata": md_json}
                    rows.append(
                        {
                            "id": _id,
                            "text": text,
                            "metadata": md,
                            "score": float(score),
                            "embedding": emb,
                        }
                    )
                except Exception as e:
                    app.logger.exception("Error fetching row from DB: %s", e)
        finally:
            conn.close()

        # 4) sort + MIN_SCORE filter
        rows.sort(key=lambda x: x["score"], reverse=True)
        top = [r for r in rows if r["score"] >= MIN_SCORE][:top_k]

        # Build the result object. Allow optional SUMMARY-compression of displayed top matches
        result = {
            "query": query,
            "top_matches": top,
            "original_query": original_query_saved
        }

        # Optional: compress top match texts for display (safe, non-destructive)
        compress_context = bool(payload.get("compress_context", False))
        try:
            compress_max_sentences = int(payload.get("compress_max_sentences", 1))
        except Exception:
            compress_max_sentences = 1

        if compress_context:
            # Prefer cached compressor helper if available, fall back to simple_compress_text
            compressor = globals().get("get_compressed_chunk")
            for m in result["top_matches"]:
                orig_text = m.get("text", "")
                try:
                    if callable(compressor):
                        # get_compressed_chunk expects (chunk_id, text, max_sentences)
                        summary = compressor(m.get("id"), orig_text, max_sentences=compress_max_sentences)
                    else:
                        summary = simple_compress_text(orig_text, max_sentences=compress_max_sentences)
                except Exception:
                    # Defensive fallback — keep original text when compression fails
                    summary = orig_text

                # Add diagnostic metadata so callers can see compression effect
                m["original_text_length"] = len(orig_text)
                m["compressed_text_length"] = len(summary)
                m["compression_ratio"] = round(len(summary) / (len(orig_text) + 1), 3)
                m["text"] = summary

        # 5) If no matches — return early with explicit answer
        if not top:
            result['answer'] = "I don't know (no relevant documents match)."
            return jsonify(result), 200

        # --- bias candidates by entity_hint (if available) ---
        if entity_hint:
            # expand the hint into multiple lexical forms and check both text + metadata
            expanded = expand_entity_hint(entity_hint)
            for r in top:
                text_low = (r.get('text') or '').lower()
                md = r.get('metadata') or {}
                meta_vals = ' '.join([str(v).lower() for v in md.values() if v])
                # support multiple forms in expanded (pipe-separated)
                match = False
                if expanded:
                    match = entity_in_text(expanded, text_low) or entity_in_text(expanded, meta_vals)
                else:
                    match = entity_in_text(entity_hint, text_low) or entity_in_text(entity_hint, meta_vals)
                r['entity_match'] = 1 if match else 0
            # push entity matches to the front (but keep score ordering inside each group)
            top.sort(key=lambda x: (x.get('entity_match', 0), x.get('score', 0.0)), reverse=True)
            result['top_matches'] = top

        # 6) If reranker requested, run it (optional)
        if use_reranker:
            try:
                # reranker should be implemented elsewhere; we assume rerank_candidates(query, top)
                # returns list of dicts with 'id', 'text', 'metadata', 'score', 'rerank_score' sorted by rerank_score desc
                reranked = rerank_candidates(query, top)  # <-- ensure you have this function defined
                if isinstance(reranked, list) and reranked:
                    # keep only top_k after rerank
                    top = reranked[:top_k]
                    result['top_matches'] = top
            except NameError:
                # if reranker not implemented, log and continue with original top
                app.logger.warning("rerank_candidates not found; skipping rerank")
            except Exception as e:
                app.logger.exception("Reranker error")
                # don't fail entire request for reranker issues -- surface a partial result
                result['reranker_error'] = str(e)

        # --- evidence scoring: compute query embedding once, score candidates, filter/sort ---
        try:
            q_emb_for_evidence = embed_texts([query])[0]
        except Exception:
            q_emb_for_evidence = None

        # compute evidence for each candidate in 'top' (or in rows if you prefer wider baseline)
        for r in top:
            try:
                ev = compute_evidence_score(query, r, q_emb=q_emb_for_evidence)
            except Exception as e:
                ev = {"S": 0.0, "L": 0.0, "M": 0, "C": 0, "evidence_score": 0.0}
            r['evidence'] = ev
            r['evidence_score'] = ev['evidence_score']

        # remove candidates flagged as contradictory (C==1)
        filtered = [r for r in top if r.get('evidence', {}).get('C', 0) == 0]

        # apply minimum evidence threshold (env var EVIDENCE_MIN)
        filtered = [r for r in filtered if r.get('evidence_score', -999) >= EVIDENCE_MIN]

        # final sort by evidence_score then original score as tie-breaker
        filtered.sort(key=lambda x: (x.get('evidence_score', 0.0), x.get('score', 0.0)), reverse=True)

        # keep top_k
        top = filtered[:top_k]
        # update result
        result['top_matches'] = top

        # 7) If not using LLM, return top matches
        if not use_llm:
            return jsonify(result), 200

        def chunk_supports_answer(query, chunk_text):
            q = query.lower()
            c = chunk_text.lower()

            # Token overlap requirement (cheap lexical signal)
            q_words = set(re.findall(r"\w+", q))
            c_words = set(re.findall(r"\w+", c))
            overlap = q_words & c_words
            if len(overlap) == 0:
                return False
            # require at least some meaningful overlap (at least 20% of query tokens)
            if len(overlap) / max(1, len(q_words)) < 0.20:
                return False

            # Simple inversion detection: if the chunk swaps subject/object roles, consider it unsupported
            if simple_inversion_penalty(query, chunk_text) < 0.2:
                return False

            # Optional embedding similarity check (weak) to guard against coincidental lexical overlap
            try:
                q_emb = embed_texts([query])[0]
                c_emb = embed_texts([chunk_text])[0]
                sim = cosine_similarity(q_emb, c_emb)
                # require at least a modest semantic alignment
                if sim < 0.20:
                    return False
            except Exception:
                # if embeddings fail, continue with lexical signals only
                pass

            # block explicit negations in candidate (e.g., 'not', 'never', ' no ') that contradict a positive-looking query
            neg_words = (" not ", " never ", " no ")
            if any(n in c for n in neg_words) and any(w in q for w in ("where", "how", "what", "who", "when", "which", "do", "does", "are", "is")):
                # conservative: if chunk contains negation and the query is asking positively, reject
                return False

            return True

        # 8) Build context and call LLM
        MAX_CONTEXT_CHARS = 4000
        # assemble context - join top texts (truncate to MAX_CONTEXT_CHARS)
        # Prepare the retrieved context for LLM (compress if requested, obey character budget)
        context_parts = []
        total = 0
        compress = bool(payload.get("compress_context", False))
        try:
            max_sentences = int(payload.get("compress_max_sentences", 2))
        except Exception:
            max_sentences = 2

        for t in top:
            txt = t["text"]
            chunk_id = t.get("id")  # may be int or str

            if compress:
                # Optionally compress chunk (using cached summary if available)
                try:
                    txt = get_compressed_chunk(chunk_id, txt, max_sentences=max_sentences)
                except Exception:
                    # fallback: use original text on compression error
                    pass

            # Enforce the overall MAX_CONTEXT_CHARS limit
            if total + len(txt) > MAX_CONTEXT_CHARS:
                remain = MAX_CONTEXT_CHARS - total
                if remain <= 0:
                    break  # Context budget exhausted
                txt = txt[:remain]  # Truncate text to fit remaining space

            context_parts.append(txt)
            total += len(txt)

        # Join context chunks with a separator for clarity
        context = "\n\n---\n\n".join(context_parts)

        system_prompt = (
            "You are a concise assistant whose only job is to answer the user's question using ONLY the "
            "provided CONTEXT. Use synonyms and paraphrases when appropriate. If the context contains "
            "an explicit factual sentence that answers the question, reply with that answer (concise, 1-2 sentences). "
            "If the context does not contain the answer, reply: \"I don't know.\" Do NOT invent extra facts. "
            "If multiple context lines have the answer, combine briefly."
        )
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

            # robust extraction
            try:
                raw_text = ai_resp.choices[0].message.content.strip()
            except Exception:
                try:
                    # fallback to dict style
                    raw_text = (ai_resp.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
                except Exception:
                    raw_text = ""

            supported_chunks = [t for t in top if chunk_supports_answer(query, t["text"])]

            # compute a conservative confidence estimate and gate answers below threshold
            try:
                support_ratio = float(len(supported_chunks)) / float(max(1, len(top)))
                top_score = float(top[0].get('score', 0.0)) if top else 0.0
                # confidence mixes evidence density (support_ratio) and strongest semantic score
                confidence = 0.5 * support_ratio + 0.5 * min(1.0, top_score)
            except Exception:
                confidence = 0.0

            if confidence < RAG_CONFIDENCE_THRESHOLD:
                # do not answer when confidence is low — be explicit and return retrieved matches for inspection
                return jsonify({
                    "query": query,
                    "original_query": original_query_saved,
                    "answer": "I don't know.",
                    "confidence": round(float(confidence), 3),
                    "reason": "Low confidence from retrieved evidence",
                    "top_matches": top
                }), 200

            if not supported_chunks:
                return jsonify({
                    "query": query,
                    "answer": "I don't know.",
                    "reason": "No retrieved chunk explicitly supports an answer.",
                    "top_matches": top
                }), 200
            # debug log raw LLM output (truncated)
            app.logger.debug("DEBUG: rag_query raw LLM response: %s", raw_text[:2000])

            answer = raw_text

            # If model explicitly says it doesn't know or returns nothing, fallback to best match
            if not answer or answer.lower().strip() in ("i don't know.", "i don't know", "unknown", "no idea", "can't answer"):
                if top and len(top) > 0:
                    fallback_text = top[0].get("text", "")
                    answer = f"Best matching document excerpt: {fallback_text}"
                else:
                    answer = "I don't know."

            result['answer'] = answer or "I don't know."
            return jsonify(result), 200

        except Exception as e:
            app.logger.exception("OpenAI LLM error in rag_query")
            return jsonify({"error": "OpenAI API error", "detail": str(e)}), 502

    except Exception as e:
        # Catch-all to ensure we always return a JSON response
        app.logger.exception("Unexpected error in rag_query")
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500

@app.route('/rewrite-query', methods=['POST'])
def rewrite_query():
    if not request.is_json:
        return jsonify({"error":"JSON body required"}), 400
    payload = request.get_json()
    query = (payload.get("query") or "").strip()
    if not query:
        return jsonify({"error":"'query' required"}), 400

    system_prompt = (
        "You will REWRITE user queries into a short, explicit search query suitable for document retrieval. "
        "If the user mentions a concrete entity (e.g., 'apple', 'Python'), normalize it (use singular/plural as appropriate). "
        "If ambiguous, TRY to infer the most-likely entity. "
        "Return ONLY a JSON object with two keys: "
        "'rewritten' (string) and 'entity' (string or null). Examples:\n"
        '{"rewritten":"Where do apples grow?","entity":"apples"}\n'
        '{"rewritten":"How to train a dog to sit?","entity":"dog"}'
    )
    user_prompt = f"Original query:\n{query}\n\nRewrite and extract entity if possible."
    try:
        raw = call_chat_with_system(system_prompt, user_prompt, max_tokens=60, temperature=0.0)
        # robust parse: try JSON first, then fallback to plain text
        try:
            parsed = json.loads(raw)
            rewritten = parsed.get("rewritten", raw).strip()
            entity = parsed.get("entity")
        except Exception:
            # fallback: take whole raw as rewritten, entity = None
            rewritten = raw.strip().splitlines()[0].strip()
            entity = None
        return jsonify({"original": query, "rewritten": rewritten, "entity": entity}), 200
    except Exception as e:
        return jsonify({"error":"LLM error","detail":str(e)}), 502
        
@app.route("/compress", methods=["POST"])
def compress_endpoint():
    """
    Endpoint for fast, deterministic text compression.

    POST JSON:
      { "text": "...", "max_sentences": 2 }
    Returns:
      { "summary": "..." }

    - 'text' (str, required): Text to compress.
    - 'max_sentences' (int, optional): Max number of sentences to keep (default: 2).

    Used for RAG context compression.
    """
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400

    payload = request.get_json()
    text = payload.get("text", "")
    if not text:
        return jsonify({"error": "'text' required"}), 400

    try:
        ms = int(payload.get("max_sentences", 2))
    except Exception:
        ms = 2  # Fallback default if conversion fails

    # Compress the text using simple_compress_text utility
    summary = simple_compress_text(text, max_sentences=ms)
    return jsonify({"summary": summary}), 200
    
@app.route("/compress-cache-clear", methods=["POST"])
def compress_cache_clear():
    """
    Endpoint to clear the in-memory compression summary cache (COMPRESS_CACHE).
    This helps reset state if summaries have changed or for testing purposes.

    Returns:
        JSON with status, cleared flag, and current cache_size (should be 0).
    """
    COMPRESS_CACHE.clear()
    return jsonify({
        "status": "ok",
        "cleared": True,
        "cache_size": len(COMPRESS_CACHE)
    }), 200
    
@app.route('/rag-debug', methods=['POST'])
def rag_debug():
    """
    Debug endpoint: run the retrieval pipeline but return full candidates and computed evidence
    without aggressive filtering. POST JSON: {"query":"...", "fetch_k": 10}
    """
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    payload = request.get_json()
    query = (payload.get('query') or '').strip()
    if not query:
        return jsonify({"error": "'query' required"}), 400
    try:
        fetch_k = int(payload.get('fetch_k', 10))
    except Exception:
        fetch_k = 10
    try:
        q_emb = embed_texts([query])[0]
    except Exception as e:
        q_emb = None
    faiss_hits = search_faiss_by_vector(q_emb, top_k=fetch_k) if q_emb is not None else []
    conn = sqlite3.connect(RAG_DB)
    cur = conn.cursor()
    rows = []
    try:
        for db_row_id, score in faiss_hits:
            try:
                cur.execute("SELECT id, text, embedding, metadata FROM docs WHERE id = ?", (db_row_id,))
                r = cur.fetchone()
                if not r:
                    continue
                _id, text, emb_json, md_json = r
                try:
                    emb = json.loads(emb_json) if emb_json else None
                except Exception:
                    emb = None
                try:
                    md = json.loads(md_json) if md_json else {}
                except Exception:
                    md = {"_raw_metadata": md_json}
                cand = {"id": _id, "text": text, "metadata": md, "score": float(score), "embedding": emb}
                try:
                    cand['evidence'] = compute_evidence_score(query, cand, q_emb=q_emb)
                except Exception:
                    cand['evidence'] = {"S": 0.0, "L": 0.0, "M": 0, "C": 0, "evidence_score": 0.0}
                rows.append(cand)
            except Exception:
                continue
    finally:
        conn.close()
    return jsonify({"query": query, "candidates": rows}), 200

    # -----------------------------
    # Minimal RAG evaluation endpoint

    from collections import Counter

    # Example golden set (replace with your own domain-specific Q/A pairs).
    # Each item should have: query (str), gold_answer (str)
    GOLD_SET = [
        {"query": "Where do apples grow?", "gold_answer": "Apples grow on trees."},
        {"query": "What is an ETF?", "gold_answer": "An ETF is an exchange-traded fund, a basket of securities traded on an exchange."},
        {"query": "How many calories in a bowl of oatmeal?", "gold_answer": "Approximately 150-200 calories depending on portion and preparation."},
        {"query": "How to improve squat form?", "gold_answer": "Keep chest up, knees tracking toes, sit back, and keep weight on heels."},
        {"query": "What is a transformer in AI?", "gold_answer": "A transformer is a neural architecture based on self-attention used for sequence modeling."},
        {"query": "What is ROE?", "gold_answer": "Return on Equity (ROE) = Net Income / Shareholders' Equity."},
        {"query": "How often should I do cardio each week?", "gold_answer": "Generally 2-4 sessions per week depending on goals and fitness level."},
        {"query": "What does 'fine-tuning' mean for models?", "gold_answer": "Fine-tuning adjusts model weights on task-specific data after pretraining."},
        {"query": "Is diversification important in investing?", "gold_answer": "Yes — it reduces single-asset risk by spreading exposure across assets."},
        {"query": "How long to rest between heavy sets?", "gold_answer": "Typically 2-5 minutes for maximal strength; 30-90s for hypertrophy."},
    ]

    # Helper: compute simple token overlap ratio between two strings
    def token_overlap_ratio(a: str, b: str) -> float:
        """
        Return the ratio of overlapping tokens in 'a' compared to the number in 'a'.
        Used as a simple similarity measure for RAG evaluation.
        """
        if not a or not b:
            return 0.0
        a_tokens = set(re.findall(r'\w+', a.lower()))
        b_tokens = set(re.findall(r'\w+', b.lower()))
        if not a_tokens:
            return 0.0
        return float(len(a_tokens & b_tokens)) / float(len(a_tokens))

    # Endpoint: run the evaluation suite
    @app.route('/rag-eval', methods=['POST'])
    def rag_eval():
        """
        Run a small automatic RAG evaluation against the current FAISS + DB index.
        POST JSON optional:
          { "gold": [ { "query": "...", "gold_answer":"..." }, ... ], "top_k": 5 }
        If 'gold' omitted, uses embedded GOLD_SET above.
        Returns per-query metrics and aggregated summary.
        """
        # Accept both POST with and without JSON payload
        if not request.is_json:
            payload = {}
        else:
            payload = request.get_json() or {}

        # Use provided gold set, or default GOLD_SET
        gold = payload.get('gold') or GOLD_SET
        try:
            top_k = int(payload.get('top_k', 5))
        except Exception:
            top_k = 5

        results = []
        # Counters for aggregate metrics
        recall_at_k_counters = {1: 0, 3: 0, 5: 0}
        support_counts = 0
        hallucination_counts = 0

        # Prewarm embedding client (optional; best-effort)
        try:
            _ = embed_texts(["warmup"])
        except Exception:
            pass

        conn = sqlite3.connect(RAG_DB)
        cur = conn.cursor()

        for item in gold:
            q = (item.get('query') or '').strip()
            gold_a = (item.get('gold_answer') or '').strip()
            if not q:
                continue

            q_emb = None
            try:
                q_emb = embed_texts([q])[0]
            except Exception as e:
                app.logger.debug("rag-eval embedding failed for query '%s': %s", q, e)
                # Embedding failed; skip this query and log error
                results.append({"query": q, "error": "embedding_failed"})
                continue

            try:
                hits = search_faiss_by_vector(q_emb, top_k=max(top_k, 5))
            except Exception as e:
                app.logger.debug("rag-eval faiss search failed for query '%s': %s", q, e)
                results.append({"query": q, "error": "faiss_failed"})
                continue

            retrieved_texts = []
            retrieved_ids = []
            retrieved_scores = []
            for db_row_id, score in hits:
                try:
                    cur.execute("SELECT id, text, metadata FROM docs WHERE id = ?", (db_row_id,))
                    r = cur.fetchone()
                    if not r:
                        continue
                    _id, text, md_json = r
                    retrieved_texts.append(text)
                    retrieved_ids.append(_id)
                    retrieved_scores.append(float(score))
                except Exception:
                    # Skip retrieval issues, keep going
                    continue

            # Compute Recall@K:
            for K in (1, 3, 5):
                top_texts = retrieved_texts[:K]
                found = False
                for t in top_texts:
                    # Conservative threshold, change as needed
                    if token_overlap_ratio(gold_a, t) >= 0.45:
                        found = True
                        break
                if found:
                    recall_at_k_counters[K] += 1

            # Compute support rate: fraction of retrieved chunks that support the answer
            supports = 0
            for t in retrieved_texts[:top_k]:
                try:
                    if chunk_supports_answer(q, t):
                        supports += 1
                except Exception:
                    # Robust to errors inside chunk_supports_answer
                    pass
            support_rate = float(supports) / max(1, min(top_k, len(retrieved_texts)))
            if support_rate > 0:
                support_counts += 1

            # Hallucination proxy: call rag_query endpoint and check for unsupported answer tokens
            llm_answer = None
            hallucinated = False
            try:
                # Use lightweight request context to call rag_query internally
                with app.test_request_context(json={
                        "query": q,
                        "top_k": top_k,
                        "use_llm": True,
                        "use_reranker": False
                    }):
                    resp = rag_query()
                    # rag_query may return tuple or Response
                    if isinstance(resp, tuple):
                        body = resp[0].get_json() if hasattr(resp[0], 'get_json') else resp[0]
                    elif hasattr(resp, 'get_json'):
                        body = resp.get_json()
                    else:
                        body = resp
                    llm_answer = (body.get('answer') or "").strip()

                    # Hallucination check: are tokens in the answer present in any retrieved_texts?
                    ans_tokens = set(re.findall(r'\w+', llm_answer.lower()))
                    retrieved_tokens = set()
                    for t in retrieved_texts:
                        retrieved_tokens.update(re.findall(r'\w+', t.lower()))
                    # Compute fraction of answer tokens not in retrieved texts
                    if len(ans_tokens) > 0:
                        missing = len([tok for tok in ans_tokens if tok not in retrieved_tokens])
                        missing_ratio = float(missing) / float(len(ans_tokens))
                        # Flag as hallucination proxy if more than 40% of answer tokens are absent
                        hallucinated = (missing_ratio > 0.40)
                        if hallucinated:
                            hallucination_counts += 1
            except Exception as e:
                app.logger.debug("rag-eval rag_query call failed for '%s': %s", q, e)

            # Collect per-query results
            results.append({
                "query": q,
                "gold_answer": gold_a,
                "retrieved_ids": retrieved_ids[:top_k],
                "retrieved_scores": retrieved_scores[:top_k],
                "support_rate": round(support_rate, 3),
                "llm_answer": llm_answer,
                "hallucination_proxy": bool(hallucinated)
            })

        # Close the DB connection
        conn.close()

        # Compose summary metrics for the response
        total = float(len(results)) if results else 1.0
        summary = {
            "num_queries": int(total),
            "recall_at_1": round(recall_at_k_counters[1] / total, 3),
            "recall_at_3": round(recall_at_k_counters[3] / total, 3),
            "recall_at_5": round(recall_at_k_counters[5] / total, 3),
            "support_positive_rate": round(support_counts / total, 3),  # Fraction of queries with at least one supporting chunk
            "hallucination_rate_proxy": round(hallucination_counts / total, 3),
        }

        return jsonify({"summary": summary, "per_query": results}), 200
# Run the server
# -----------------------------
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5050, debug=True)