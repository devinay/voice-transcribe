import pathlib

from voice_transcribe import storage


def test_process_and_save_writes_markdown_and_indexes(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "TRANSCRIPT_DIR", tmp_path)
    monkeypatch.setattr(storage, "load_process_prompt", lambda transcript: f"PROMPT::{transcript}")

    calls = []

    def fake_run_llm_prompt(prompt, llm_backend, ollama_model):
        calls.append((prompt, llm_backend, ollama_model))
        if len(calls) == 1:
            return "# Title\n\n## Summary\nSome summary.\n\n## Source Notes\n<details><summary>Original Notes</summary>\nraw transcript\n</details>\n\n## Processed Transcript\nHello.\n\n## Actions / Follow-ups\n- [ ] Test\n"
        return "daily standup summary draft v1"

    indexed_paths = []

    monkeypatch.setattr(storage, "run_llm_prompt", fake_run_llm_prompt)
    monkeypatch.setattr(storage.vector, "on_doc_saved", lambda p: indexed_paths.append(pathlib.Path(p)))

    session_dir = storage.process_and_save("raw transcript", "ollama", "qwen2.5:7b-instruct")

    name = "daily_standup_summary_draft_v1"
    assert session_dir == tmp_path / name
    saved = session_dir / f"{name}.md"
    assert saved.exists()
    assert "## Summary" in saved.read_text()

    assert len(calls) == 2
    assert calls[0][1:] == ("ollama", "qwen2.5:7b-instruct")
    assert calls[1][1:] == ("ollama", "qwen2.5:7b-instruct")
    assert indexed_paths == [saved]


def test_update_session_artifact_appends_generated_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "TRANSCRIPT_DIR", tmp_path)
    monkeypatch.setattr(
        storage,
        "run_llm_prompt",
        lambda prompt, llm_backend, ollama_model: (
            "# Title\n\n## Summary\nSome summary.\n\n## Source Notes\n<details><summary>Original Notes</summary>\nhello\n</details>\n\n## Processed Transcript\nHello.\n\n## Actions / Follow-ups\n"
        ),
    )

    session_dir = storage.start_session_artifact()
    md_path = storage.update_session_artifact(
        session_dir=session_dir,
        source_notes="hello",
        llm_backend="ollama",
        ollama_model="qwen2.5:7b-instruct",
        generated_artifacts=[
            {
                "kind": "diagram",
                "title": "Diagram 1",
                "description": "A simple diagram",
                "path": "diagram_1.png",
            }
        ],
    )

    text = md_path.read_text()
    assert "## Generated Artifacts" in text
    assert "![Diagram 1](diagram_1.png)" in text
