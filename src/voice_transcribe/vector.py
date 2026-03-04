"""Local vector store for transcript summaries.

Uses LanceDB for storage and sentence-transformers for embeddings.
Only the ## Summary section of each document is embedded — never headers,
raw transcript, or next-actions content.

See PLAN.md § Phase 2 for the full specification.
"""

import colorsys
import hashlib
import pathlib
import re
import sys
import time
from collections import Counter

import pyarrow as pa

from voice_transcribe.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    VECTOR_DB_DIR,
    VECTOR_SIMILARITY_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Color palette — 64 visually distinct colors, alternating lightness
# ---------------------------------------------------------------------------

def _build_palette(n: int = 64) -> list[str]:
    palette = []
    for i in range(n):
        h = i / n
        l = 0.40 if i % 2 == 0 else 0.58
        r, g, b = colorsys.hls_to_rgb(h, l, 0.75)
        palette.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return palette


PALETTE: list[str] = _build_palette()

# ---------------------------------------------------------------------------
# LanceDB schema
# ---------------------------------------------------------------------------

_SCHEMA = pa.schema([
    pa.field("doc_id",             pa.string()),
    pa.field("path",               pa.string()),
    pa.field("created_at",         pa.string()),
    pa.field("updated_at",         pa.string()),
    pa.field("embedding_model",    pa.string()),
    pa.field("dictionary_version", pa.int32()),
    pa.field("color_hex",          pa.string()),
    pa.field("color_index",        pa.int32()),
    pa.field("vector",             pa.list_(pa.float32(), EMBEDDING_DIM)),
])

# ---------------------------------------------------------------------------
# Lazy singletons (model + DB opened on first use)
# ---------------------------------------------------------------------------

_table = None
_embed_model = None


def _get_table():
    global _table
    if _table is not None:
        return _table
    import lancedb
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(VECTOR_DB_DIR))
    _table = db.create_table("transcripts", schema=_SCHEMA, exist_ok=True)
    return _table


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _embed(text: str) -> list[float]:
    return _get_embed_model().encode(text, normalize_embeddings=True).tolist()


def extract_summary(path: pathlib.Path) -> str:
    """Extract the ## Summary section from a processed transcript markdown file.

    Matches content between '## Summary' and the next '##' header or
    '<details' block, whichever comes first.
    """
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"##\s+Summary\s*\n(.*?)(?=\n##\s|\n<details|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return ""
    return match.group(1).strip()


def _find_by_doc_id(table, doc_id: str) -> dict | None:
    """Return the existing row for doc_id, or None."""
    try:
        df = table.to_pandas()
        rows = df[df["doc_id"] == doc_id]
        if len(rows) == 0:
            return None
        row = rows.iloc[0].to_dict()
        row["color_index"] = int(row["color_index"])
        row["dictionary_version"] = int(row["dictionary_version"])
        return row
    except Exception:
        return None


def _assign_color(table, embedding: list[float], doc_id: str) -> tuple[int, str]:
    """Return (color_index, color_hex) for a new document.

    If a neighbour with similarity >= threshold exists (excluding same doc_id),
    reuse its color. Otherwise assign the least-used palette color.
    """
    try:
        results = (
            table.search(embedding)
            .metric("cosine")
            .where(f"doc_id != '{doc_id}'")
            .limit(1)
            .to_list()
        )
        if results:
            # LanceDB cosine: _distance = 1 - cosine_similarity (normalised vecs)
            similarity = 1.0 - results[0].get("_distance", 1.0)
            if similarity >= VECTOR_SIMILARITY_THRESHOLD:
                return int(results[0]["color_index"]), str(results[0]["color_hex"])
    except Exception:
        pass  # empty table or search error — fall through

    # Assign the least-used palette slot
    try:
        df = table.to_pandas()
        counts = Counter(df["color_index"].tolist()) if len(df) > 0 else {}
    except Exception:
        counts = {}

    best = min(range(len(PALETTE)), key=lambda i: counts.get(i, 0))
    return best, PALETTE[best]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def on_doc_saved(path: pathlib.Path) -> None:
    """Index a saved transcript document.

    Called from storage.process_and_save after each successful save.
    Extracts the ## Summary section, embeds it, assigns a similarity color,
    and upserts the record into LanceDB. Silent on success.
    """
    summary = extract_summary(path)
    if not summary:
        return

    doc_id = _sha256(summary)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")

    try:
        table = _get_table()
        embedding = _embed(summary)
        existing = _find_by_doc_id(table, doc_id)

        if existing:
            # Same summary content: keep existing color, update path + timestamp
            color_index = existing["color_index"]
            color_hex = existing["color_hex"]
            created_at = existing["created_at"]
        else:
            color_index, color_hex = _assign_color(table, embedding, doc_id)
            created_at = now

        table.merge_insert("doc_id") \
            .when_matched_update_all() \
            .when_not_matched_insert_all() \
            .execute([{
                "doc_id":             doc_id,
                "path":               str(path),
                "created_at":         created_at,
                "updated_at":         now,
                "embedding_model":    EMBEDDING_MODEL,
                "dictionary_version": 0,
                "color_hex":          color_hex,
                "color_index":        color_index,
                "vector":             embedding,
            }])

    except Exception as e:
        print(f"\n[vector] indexing failed: {e}", file=sys.stderr)


def search(
    query_text: str,
    top_k: int = 5,
    threshold: float = VECTOR_SIMILARITY_THRESHOLD,
) -> list[dict]:
    """Find documents similar to query_text.

    Returns list of dicts with keys: doc_id, path, color_hex, color_index, similarity.
    Results are filtered to similarity >= threshold.
    """
    embedding = _embed(query_text)
    try:
        results = (
            _get_table()
            .search(embedding)
            .metric("cosine")
            .limit(top_k)
            .to_list()
        )
    except Exception:
        return []

    out = []
    for r in results:
        similarity = 1.0 - r.get("_distance", 1.0)
        if similarity >= threshold:
            out.append({
                "doc_id":      r["doc_id"],
                "path":        r["path"],
                "color_hex":   r["color_hex"],
                "color_index": int(r["color_index"]),
                "similarity":  round(similarity, 4),
            })
    return out
