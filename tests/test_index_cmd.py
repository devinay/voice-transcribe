"""Tests for voice_transcribe.index_cmd (voice-index CLI)."""

import pytest

import voice_transcribe.index_cmd as index_cmd
import voice_transcribe.vector as vmod


@pytest.fixture(autouse=True)
def reset_singletons():
    vmod._table = None
    vmod._embed_model = None
    yield
    vmod._table = None
    vmod._embed_model = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(monkeypatch, argv=None):
    monkeypatch.setattr("sys.argv", argv or ["voice-index"])
    index_cmd.main()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_md_files_exits_zero(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(index_cmd, "TRANSCRIPT_DIR", tmp_path)
    monkeypatch.setattr("sys.argv", ["voice-index"])

    with pytest.raises(SystemExit) as exc:
        index_cmd.main()

    assert exc.value.code == 0
    assert "No .md files found" in capsys.readouterr().out


def test_indexes_all_md_files(tmp_path, monkeypatch, capsys):
    (tmp_path / "a.md").write_text("# A\n\n## Summary\nSummary A.\n")
    (tmp_path / "b.md").write_text("# B\n\n## Summary\nSummary B.\n")

    indexed = []
    monkeypatch.setattr(index_cmd.vector, "on_doc_saved", lambda p: indexed.append(p))
    monkeypatch.setattr("sys.argv", ["voice-index"])
    monkeypatch.setattr(index_cmd, "TRANSCRIPT_DIR", tmp_path)

    index_cmd.main()

    assert len(indexed) == 2
    out = capsys.readouterr().out
    assert "2 indexed" in out
    assert "0 failed" in out


def test_continues_after_single_failure(tmp_path, monkeypatch, capsys):
    (tmp_path / "good.md").write_text("# Good\n\n## Summary\nGood summary.\n")
    (tmp_path / "bad.md").write_text("# Bad\n\n## Summary\nBad summary.\n")

    def fail_on_bad(p):
        if "bad" in p.name:
            raise RuntimeError("forced failure")

    monkeypatch.setattr(index_cmd.vector, "on_doc_saved", fail_on_bad)
    monkeypatch.setattr("sys.argv", ["voice-index"])
    monkeypatch.setattr(index_cmd, "TRANSCRIPT_DIR", tmp_path)

    index_cmd.main()  # must not raise

    out = capsys.readouterr().out
    assert "1 indexed" in out
    assert "1 failed" in out


def test_custom_dir_via_flag(tmp_path, monkeypatch, capsys):
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    (custom_dir / "c.md").write_text("# C\n\n## Summary\nCustom summary.\n")

    indexed = []
    monkeypatch.setattr(index_cmd.vector, "on_doc_saved", lambda p: indexed.append(p))
    monkeypatch.setattr("sys.argv", ["voice-index", "--dir", str(custom_dir)])

    index_cmd.main()

    assert len(indexed) == 1
    assert "1 indexed" in capsys.readouterr().out


def test_ignores_non_md_files(tmp_path, monkeypatch, capsys):
    (tmp_path / "doc.md").write_text("# D\n\n## Summary\nReal doc.\n")
    (tmp_path / "notes.txt").write_text("plain text, not markdown")
    (tmp_path / "data.json").write_text('{"key": "value"}')

    indexed = []
    monkeypatch.setattr(index_cmd.vector, "on_doc_saved", lambda p: indexed.append(p))
    monkeypatch.setattr("sys.argv", ["voice-index"])
    monkeypatch.setattr(index_cmd, "TRANSCRIPT_DIR", tmp_path)

    index_cmd.main()

    assert len(indexed) == 1  # only the .md file
