import pathlib

from voice_transcribe import storage


def test_process_and_save_writes_markdown_and_indexes(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "TRANSCRIPT_DIR", tmp_path)
    monkeypatch.setattr(storage, "load_process_prompt", lambda transcript: f"PROMPT::{transcript}")

    calls = []

    def fake_run_llm_prompt(prompt, llm_backend, ollama_model):
        calls.append((prompt, llm_backend, ollama_model))
        if len(calls) == 1:
            return "# Title\n\n## Summary\nSome summary.\n"
        return "daily standup summary draft v1"

    indexed_paths = []

    monkeypatch.setattr(storage, "run_llm_prompt", fake_run_llm_prompt)
    monkeypatch.setattr(storage.vector, "on_doc_saved", lambda p: indexed_paths.append(pathlib.Path(p)))

    storage.process_and_save("raw transcript", "ollama", "qwen2.5:7b-instruct")

    saved = tmp_path / "daily_standup_summary_draft_v1.md"
    assert saved.exists()
    assert "## Summary" in saved.read_text()

    assert len(calls) == 2
    assert calls[0][1:] == ("ollama", "qwen2.5:7b-instruct")
    assert calls[1][1:] == ("ollama", "qwen2.5:7b-instruct")
    assert indexed_paths == [saved]
