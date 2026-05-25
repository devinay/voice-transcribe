"""Markdown-centered conversation session runner.

This loop keeps a live `session.md` artifact on disk as the durable base of the
conversation. Only user speech is used to build that artifact. Agent/tool
outputs are embedded back into the markdown as derived artifacts.
"""
from __future__ import annotations

import pathlib

from rich.console import Console
from rich.markdown import Markdown

from voice_transcribe.agents import diagram as diagram_agent
from voice_transcribe.deepgram_stream import run_push_to_talk_session
from voice_transcribe.llm import run_llm_prompt
from voice_transcribe.storage import (
    finalize_session_artifact,
    start_session_artifact,
    update_session_artifact,
)

_BOLD = "\033[1m"
_RESET = "\033[0m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"

_console = Console()

_ASSISTANT_PROMPT = """You are helping the user iterate on a markdown session artifact.

Rules:
- The markdown is the durable base artifact and is built only from the user's spoken content.
- Respond in plain text only.
- Keep replies short and useful.
- Ask at most one follow-up question at a time.
- If a diagram or other follow-up action would help, suggest it plainly.
- Do not output JSON, markdown fences, or tool syntax.
- Do not restate the whole markdown back to the user.

Current session markdown:
{markdown}

Latest user turn:
{user_turn}
"""


def run(
    api_key: str,
    deepgram_model: str,
    llm_backend: str,
    ollama_model: str,
) -> None:
    session_dir = start_session_artifact()
    source_turns: list[str] = []
    generated_artifacts: list[dict] = []
    md_path = session_dir / "session.md"

    print("  Starting conversation. Ctrl+C to finalize and exit.\n")
    print(f"  {_DIM}live session → {session_dir}{_RESET}\n", flush=True)

    while True:
        try:
            user_text = run_push_to_talk_session(api_key=api_key, model=deepgram_model)
        except KeyboardInterrupt:
            final_dir = _finalize_session(session_dir, llm_backend, ollama_model)
            _display_markdown(final_dir / f"{final_dir.name}.md")
            print(f"\n  {_GREEN}Session saved →{_RESET} {final_dir}", flush=True)
            break

        user_text = user_text.strip()
        if not user_text:
            print("  (no speech detected)", flush=True)
            continue

        print(f"\n  You: {user_text}\n", flush=True)

        if _is_finish_request(user_text):
            final_dir = _finalize_session(session_dir, llm_backend, ollama_model)
            _display_markdown(final_dir / f"{final_dir.name}.md")
            print(f"\n  {_GREEN}Session saved →{_RESET} {final_dir}", flush=True)
            break

        source_turns.append(user_text)
        md_path = update_session_artifact(
            session_dir=session_dir,
            source_notes="\n\n".join(source_turns),
            llm_backend=llm_backend,
            ollama_model=ollama_model,
            generated_artifacts=generated_artifacts,
        )

        if _is_diagram_request(user_text):
            _handle_diagram_request(
                session_dir=session_dir,
                md_path=md_path,
                user_text=user_text,
                llm_backend=llm_backend,
                ollama_model=ollama_model,
                source_turns=source_turns,
                generated_artifacts=generated_artifacts,
            )
            continue

        assistant_message = _assistant_reply(md_path, user_text, llm_backend, ollama_model)
        print(f"  {_BOLD}Assistant:{_RESET} {assistant_message}\n", flush=True)


def _assistant_reply(
    md_path: pathlib.Path,
    user_text: str,
    llm_backend: str,
    ollama_model: str,
) -> str:
    markdown = md_path.read_text() if md_path.exists() else ""
    prompt = _ASSISTANT_PROMPT.format(markdown=markdown, user_turn=user_text)
    return run_llm_prompt(prompt, llm_backend, ollama_model).strip()


def _handle_diagram_request(
    session_dir: pathlib.Path,
    md_path: pathlib.Path,
    user_text: str,
    llm_backend: str,
    ollama_model: str,
    source_turns: list[str],
    generated_artifacts: list[dict],
) -> None:
    diagram_index = sum(1 for artifact in generated_artifacts if artifact.get("kind") == "diagram") + 1
    output_path = session_dir / f"diagram_{diagram_index}.png"
    description = _build_diagram_description(md_path, user_text)

    print(f"  {_YELLOW}Generating diagram from session markdown...{_RESET}", flush=True)
    result = diagram_agent.run(
        description=description,
        output_path=output_path,
        llm_backend=llm_backend,
        ollama_model=ollama_model,
    )

    if not result.get("success"):
        print(f"  [diagram error] {result.get('error', 'unknown error')}", flush=True)
        return

    generated_artifacts.append(
        {
            "kind": "diagram",
            "title": f"Diagram {diagram_index}",
            "description": user_text,
            "path": output_path.name,
        }
    )
    updated_md = update_session_artifact(
        session_dir=session_dir,
        source_notes="\n\n".join(source_turns),
        llm_backend=llm_backend,
        ollama_model=ollama_model,
        generated_artifacts=generated_artifacts,
    )

    print(f"  {_GREEN}Diagram saved →{_RESET} {output_path}", flush=True)
    _display_markdown(updated_md)
    print(
        f"\n  {_BOLD}Assistant:{_RESET} I embedded the diagram into the session markdown. "
        "If you want changes, describe the edit and mention the diagram in your next turn.\n",
        flush=True,
    )


def _build_diagram_description(md_path: pathlib.Path, user_text: str) -> str:
    markdown = md_path.read_text() if md_path.exists() else ""
    return (
        "Use the markdown session artifact below as the source of truth. "
        "Create or update a diagram that matches the user's request.\n\n"
        f"Session markdown:\n{markdown}\n\n"
        f"User's latest diagram request:\n{user_text}\n"
    )


def _finalize_session(session_dir: pathlib.Path, llm_backend: str, ollama_model: str) -> pathlib.Path:
    return finalize_session_artifact(session_dir, llm_backend, ollama_model)


def _display_markdown(md_path: pathlib.Path) -> None:
    if not md_path.exists():
        return
    print()
    _console.rule("Session Markdown")
    _console.print(Markdown(md_path.read_text()))
    print()


def _is_finish_request(user_text: str) -> bool:
    lowered = user_text.lower()
    finish_phrases = (
        "finish this session",
        "save and exit",
        "save this",
        "we're done",
        "we are done",
        "finalize this",
    )
    return any(phrase in lowered for phrase in finish_phrases)


def _is_diagram_request(user_text: str) -> bool:
    lowered = user_text.lower()
    keywords = ("diagram", "draw", "visualize", "flowchart", "sequence diagram", "architecture diagram")
    return any(keyword in lowered for keyword in keywords)
