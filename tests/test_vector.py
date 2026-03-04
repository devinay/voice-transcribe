"""Tests for voice_transcribe.vector (Phase 2).

Strategy:
- Pure functions (extract_summary, palette, _sha256): tested directly, no mocking.
- DB-backed functions: use a real LanceDB in tmp_path to avoid mock drift.
  The `tmp_db` fixture patches VECTOR_DB_DIR to an isolated tmp dir.
- Embedding: `_embed` is monkeypatched to return deterministic unit vectors
  so the model is never loaded during tests.
- Singletons (_table, _embed_model) are reset before every test via autouse fixture.
"""

import hashlib
import re

import pytest

import voice_transcribe.vector as vmod
from voice_transcribe.vector import PALETTE, _sha256, extract_summary

# ---------------------------------------------------------------------------
# Reusable unit vectors (384-dim, L2-normalised)
# VEC_X · VEC_X = 1.0  (similarity 1.0, always above 0.82 threshold)
# VEC_X · VEC_Y = 0.0  (similarity 0.0, always below 0.82 threshold)
# ---------------------------------------------------------------------------

VEC_X = [1.0] + [0.0] * 383
VEC_Y = [0.0, 1.0] + [0.0] * 382


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset lazy singletons before and after every test."""
    vmod._table = None
    vmod._embed_model = None
    yield
    vmod._table = None
    vmod._embed_model = None


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """Point the vector store at a fresh isolated LanceDB directory."""
    monkeypatch.setattr(vmod, "VECTOR_DB_DIR", tmp_path / "index.lancedb")
    return tmp_path


@pytest.fixture()
def fixed_embed(monkeypatch):
    """Replace _embed with a hash-based deterministic function (no model load)."""
    def fake_embed(text: str) -> list[float]:
        # Build a 384-dim unit vector from MD5 of text
        raw = hashlib.md5(text.encode()).digest()  # 16 bytes
        vec = [(b / 127.5 - 1.0) for b in raw] * 24  # 384 values
        norm = sum(x ** 2 for x in vec) ** 0.5
        return [x / norm for x in vec]

    monkeypatch.setattr(vmod, "_embed", fake_embed)
    return fake_embed


# ---------------------------------------------------------------------------
# Helper: insert a row directly into the table
# ---------------------------------------------------------------------------

def _insert(table, doc_id, path, vector, color_index, color_hex):
    table.merge_insert("doc_id") \
        .when_matched_update_all() \
        .when_not_matched_insert_all() \
        .execute([{
            "doc_id":             doc_id,
            "path":               path,
            "created_at":         "2026-01-01T00:00:00",
            "updated_at":         "2026-01-01T00:00:00",
            "embedding_model":    "all-MiniLM-L6-v2",
            "dictionary_version": 0,
            "color_hex":          color_hex,
            "color_index":        color_index,
            "vector":             vector,
        }])


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

def test_palette_has_64_entries():
    assert len(PALETTE) == 64


def test_palette_all_valid_hex():
    for color in PALETTE:
        assert re.match(r"^#[0-9a-f]{6}$", color), f"Invalid color: {color}"


def test_palette_all_unique():
    assert len(set(PALETTE)) == 64


# ---------------------------------------------------------------------------
# _sha256
# ---------------------------------------------------------------------------

def test_sha256_is_deterministic():
    assert _sha256("hello") == _sha256("hello")


def test_sha256_differs_for_different_inputs():
    assert _sha256("hello") != _sha256("world")


# ---------------------------------------------------------------------------
# extract_summary
# ---------------------------------------------------------------------------

def test_extract_summary_basic(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text(
        "# Title\n\n## Summary\nThis is the summary.\n\n## Next Actions\n- [ ] Do thing\n"
    )
    assert extract_summary(f) == "This is the summary."


def test_extract_summary_stops_at_next_header(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text(
        "# Title\n\n## Summary\nSummary line.\n\n## Processed Transcript\nCleaned text.\n"
    )
    assert extract_summary(f) == "Summary line."


def test_extract_summary_stops_before_details_block(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text(
        "# Title\n\n## Summary\nSummary text.\n\n<details>\n<summary>Original</summary>\nraw\n</details>\n"
    )
    assert extract_summary(f) == "Summary text."


def test_extract_summary_case_insensitive(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# Title\n\n## summary\nLowercase header.\n\n## Next Actions\n")
    assert extract_summary(f) == "Lowercase header."


def test_extract_summary_is_last_section(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# Title\n\n## Summary\nFinal section, no following header.\n")
    assert extract_summary(f) == "Final section, no following header."


def test_extract_summary_missing_returns_empty(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# Title\n\nNo summary section here.\n")
    assert extract_summary(f) == ""


def test_extract_summary_extra_whitespace_in_header(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# Title\n\n##  Summary  \n\nSummary with leading blank line.\n\n## Next Actions\n")
    assert extract_summary(f) == "Summary with leading blank line."


def test_extract_summary_multiline(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text(
        "# Title\n\n## Summary\nLine one.\nLine two.\nLine three.\n\n## Next Actions\n"
    )
    result = extract_summary(f)
    assert "Line one." in result
    assert "Line three." in result


# ---------------------------------------------------------------------------
# _assign_color
# ---------------------------------------------------------------------------

def test_assign_color_empty_table_picks_index_zero(tmp_db):
    table = vmod._get_table()
    idx, hex_color = vmod._assign_color(table, VEC_X, "any_id")
    assert idx == 0
    assert hex_color == PALETTE[0]


def test_assign_color_reuses_color_for_similar_doc(tmp_db):
    # Insert a doc with VEC_X and a specific color
    table = vmod._get_table()
    _insert(table, "existing_id", "/path/a.md", VEC_X, 42, "#abcdef")

    # Query with the same vector (cosine similarity = 1.0 > 0.82)
    idx, hex_color = vmod._assign_color(table, VEC_X, "new_id")

    assert idx == 42
    assert hex_color == "#abcdef"


def test_assign_color_dissimilar_doc_gets_new_slot(tmp_db):
    # Insert a doc with VEC_X (color_index=42)
    table = vmod._get_table()
    _insert(table, "existing_id", "/path/a.md", VEC_X, 42, "#abcdef")

    # Query with VEC_Y (orthogonal, similarity=0.0 < 0.82) → new slot
    idx, hex_color = vmod._assign_color(table, VEC_Y, "new_id")

    # Should not reuse 42; picks least-used slot (0, since 42 is in use)
    assert idx == 0
    assert hex_color == PALETTE[0]


def test_assign_color_excludes_same_doc_id(tmp_db):
    # Insert one doc; then call _assign_color with the same doc_id
    # Self should be excluded → no neighbor found → new slot assigned
    table = vmod._get_table()
    _insert(table, "same_id", "/path/a.md", VEC_X, 42, "#abcdef")

    idx, hex_color = vmod._assign_color(table, VEC_X, "same_id")

    # "same_id" excluded: no neighbor, picks least-used slot (0)
    assert idx == 0


def test_assign_color_picks_least_used_slot(tmp_db):
    # Fill slot 0 and 1; slot 2 should be picked next
    table = vmod._get_table()
    _insert(table, "doc_a", "/a.md", VEC_Y, 0, PALETTE[0])
    _insert(table, "doc_b", "/b.md", VEC_Y, 1, PALETTE[1])

    # VEC_Y is orthogonal to VEC_X → no similar neighbor → least-used
    idx, _ = vmod._assign_color(table, VEC_X, "new_id")
    assert idx == 2


# ---------------------------------------------------------------------------
# on_doc_saved
# ---------------------------------------------------------------------------

def test_on_doc_saved_skips_file_without_summary(tmp_path, tmp_db, fixed_embed):
    f = tmp_path / "no_summary.md"
    f.write_text("# Title\n\nNo summary section.\n")

    vmod.on_doc_saved(f)

    df = vmod._get_table().to_pandas()
    assert len(df) == 0


def test_on_doc_saved_inserts_new_doc(tmp_path, tmp_db, fixed_embed):
    f = tmp_path / "doc.md"
    f.write_text("# Title\n\n## Summary\nTest summary content.\n\n## Next Actions\n")

    vmod.on_doc_saved(f)

    df = vmod._get_table().to_pandas()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["path"] == str(f)
    assert row["doc_id"] == _sha256("Test summary content.")
    assert row["dictionary_version"] == 0
    assert row["color_hex"] in PALETTE
    assert 0 <= int(row["color_index"]) < 64


def test_on_doc_saved_reindex_preserves_color_and_created_at(tmp_path, tmp_db, fixed_embed):
    f = tmp_path / "doc.md"
    f.write_text("# Title\n\n## Summary\nSame summary content.\n\n## Next Actions\n")

    vmod.on_doc_saved(f)
    df1 = vmod._get_table().to_pandas()
    original_color = df1.iloc[0]["color_hex"]
    original_created = df1.iloc[0]["created_at"]

    # Re-index the same file
    vmod.on_doc_saved(f)
    df2 = vmod._get_table().to_pandas()

    assert len(df2) == 1  # no duplicate row
    assert df2.iloc[0]["color_hex"] == original_color
    assert df2.iloc[0]["created_at"] == original_created


def test_on_doc_saved_rename_updates_path(tmp_path, tmp_db, fixed_embed):
    f1 = tmp_path / "original.md"
    f2 = tmp_path / "renamed.md"
    content = "# Title\n\n## Summary\nContent stays the same.\n\n## Next Actions\n"
    f1.write_text(content)
    f2.write_text(content)  # identical summary → same doc_id

    vmod.on_doc_saved(f1)
    vmod.on_doc_saved(f2)

    df = vmod._get_table().to_pandas()
    assert len(df) == 1          # same doc_id: upserted, not duplicated
    assert df.iloc[0]["path"] == str(f2)   # path updated to latest


def test_on_doc_saved_two_distinct_docs_both_indexed(tmp_path, tmp_db, fixed_embed):
    f1 = tmp_path / "a.md"
    f2 = tmp_path / "b.md"
    f1.write_text("# A\n\n## Summary\nFirst summary.\n\n## Next Actions\n")
    f2.write_text("# B\n\n## Summary\nSecond summary.\n\n## Next Actions\n")

    vmod.on_doc_saved(f1)
    vmod.on_doc_saved(f2)

    df = vmod._get_table().to_pandas()
    assert len(df) == 2


def test_on_doc_saved_does_not_raise_on_error(tmp_path, monkeypatch, fixed_embed):
    def bad_get_table():
        raise RuntimeError("DB unavailable")

    monkeypatch.setattr(vmod, "_get_table", bad_get_table)

    f = tmp_path / "doc.md"
    f.write_text("# Title\n\n## Summary\nSome summary.\n")

    vmod.on_doc_saved(f)  # must not raise


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def test_search_returns_similar_results(tmp_db, monkeypatch):
    table = vmod._get_table()
    _insert(table, "doc1", "/path/doc1.md", VEC_X, 0, PALETTE[0])

    # Patch _embed to return the same vector → similarity = 1.0
    monkeypatch.setattr(vmod, "_embed", lambda text: VEC_X)

    results = vmod.search("any query")

    assert len(results) == 1
    assert results[0]["doc_id"] == "doc1"
    assert results[0]["path"] == "/path/doc1.md"
    assert results[0]["similarity"] >= 0.82


def test_search_filters_below_threshold(tmp_db, monkeypatch):
    table = vmod._get_table()
    _insert(table, "doc1", "/path/doc1.md", VEC_X, 0, PALETTE[0])

    # VEC_Y is orthogonal to VEC_X → similarity ≈ 0 < 0.82
    monkeypatch.setattr(vmod, "_embed", lambda text: VEC_Y)

    results = vmod.search("any query")
    assert results == []


def test_search_empty_table_returns_empty(tmp_db, monkeypatch):
    monkeypatch.setattr(vmod, "_embed", lambda text: VEC_X)
    results = vmod.search("any query")
    assert results == []


def test_search_respects_top_k(tmp_db, monkeypatch):
    table = vmod._get_table()
    # Insert 3 identical docs with different doc_ids
    _insert(table, "doc1", "/p/1.md", VEC_X, 0, PALETTE[0])
    _insert(table, "doc2", "/p/2.md", VEC_X, 1, PALETTE[1])
    _insert(table, "doc3", "/p/3.md", VEC_X, 2, PALETTE[2])

    monkeypatch.setattr(vmod, "_embed", lambda text: VEC_X)

    results = vmod.search("any query", top_k=2)
    assert len(results) <= 2
