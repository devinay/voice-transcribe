import pathlib
import re
import shutil
import time

from voice_transcribe import vector
from voice_transcribe.config import TRANSCRIPT_DIR
from voice_transcribe.llm import run_llm_prompt
from voice_transcribe.prompts import load_process_prompt

_RED = "\033[31m"
_RESET = "\033[0m"
_CLEAR = "\r" + " " * 40 + "\r"


def start_session_artifact() -> pathlib.Path:
    """Create a durable live session directory and return it."""
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = TRANSCRIPT_DIR / f"session_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def update_session_artifact(
    session_dir: pathlib.Path,
    source_notes: str,
    llm_backend: str,
    ollama_model: str,
    generated_artifacts: list[dict] | None = None,
) -> pathlib.Path:
    """Refresh the live session markdown from user source notes plus artifacts."""
    generated_artifacts = generated_artifacts or []
    markdown = _build_base_markdown(source_notes, llm_backend, ollama_model)
    artifact_section = _render_generated_artifacts(generated_artifacts)
    final_markdown = markdown.rstrip() + "\n\n" + artifact_section
    md_path = session_dir / "session.md"
    md_path.write_text(final_markdown)
    return md_path


def finalize_session_artifact(
    session_dir: pathlib.Path,
    llm_backend: str,
    ollama_model: str,
) -> pathlib.Path:
    """Rename a live session directory into a stable final transcript directory."""
    md_path = session_dir / "session.md"
    if not md_path.exists():
        return session_dir

    final_name = _generate_session_name(md_path.read_text(), llm_backend, ollama_model)
    target_dir = _unique_session_dir(TRANSCRIPT_DIR / final_name)

    if session_dir != target_dir:
        shutil.move(str(session_dir), target_dir)
    else:
        target_dir.mkdir(parents=True, exist_ok=True)

    final_md = target_dir / f"{target_dir.name}.md"
    session_md = target_dir / "session.md"
    if session_md.exists():
        shutil.move(str(session_md), final_md)

    vector.on_doc_saved(final_md)
    return target_dir


def process_and_save(transcript: str, llm_backend: str, ollama_model: str) -> pathlib.Path:
    """Compatibility helper: create, update, and finalize a single-shot session."""
    print(f"\r{_RED}●{_RESET} PROCESSING...", end="", flush=True)
    session_dir = start_session_artifact()
    update_session_artifact(session_dir, transcript, llm_backend, ollama_model)
    final_dir = finalize_session_artifact(session_dir, llm_backend, ollama_model)
    print(_CLEAR, end="", flush=True)
    print(f"\nSaved → {final_dir / (final_dir.name + '.md')}")
    return final_dir


def _build_base_markdown(source_notes: str, llm_backend: str, ollama_model: str) -> str:
    process_prompt = load_process_prompt(source_notes)
    return run_llm_prompt(process_prompt, llm_backend, ollama_model).strip() + "\n"


def _render_generated_artifacts(generated_artifacts: list[dict]) -> str:
    lines = ["## Generated Artifacts"]
    if not generated_artifacts:
        return "\n".join(lines) + "\n"

    for idx, artifact in enumerate(generated_artifacts, start=1):
        kind = artifact.get("kind", "artifact")
        title = artifact.get("title") or f"{kind.title()} {idx}"
        description = artifact.get("description", "").strip()
        path = artifact.get("path", "").strip()

        lines.append(f"### {title}")
        if description:
            lines.append(description)
        if kind == "diagram" and path:
            lines.append(f"![{title}]({path})")
        elif path:
            lines.append(f"- `{path}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _generate_session_name(markdown: str, llm_backend: str, ollama_model: str) -> str:
    filename_prompt = (
        "Output only a 5-word filename summary for the text below: "
        "lowercase words separated by underscores, no file extension, no punctuation, nothing else.\n\n"
        f"{markdown}\n"
    )
    raw = run_llm_prompt(filename_prompt, llm_backend, ollama_model).lower()
    cleaned = re.sub(r"[^a-z0-9_]+", "_", raw)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or f"session_{time.strftime('%Y%m%d_%H%M%S')}"


def _unique_session_dir(base_dir: pathlib.Path) -> pathlib.Path:
    if not base_dir.exists():
        return base_dir

    counter = 2
    while True:
        candidate = base_dir.with_name(f"{base_dir.name}_{counter}")
        if not candidate.exists():
            return candidate
        counter += 1
