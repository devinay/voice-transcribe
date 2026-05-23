import argparse
import datetime
import json
import os
import pathlib
import re
import subprocess
import sys

from voice_transcribe.audio import clear_screen
from voice_transcribe.config import (
    DEFAULT_GRANITE_MODEL,
    DEFAULT_LLM_BACKEND,
    DEFAULT_MLX_MODEL,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_MODEL,
    LOG_FILE,
)


class _StderrTee:
    """Mirrors stderr to both the terminal and a log file."""

    def __init__(self, log_path: pathlib.Path) -> None:
        self._orig = sys.stderr
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(log_path, "a", buffering=1)
        self._log.write(f"\n--- {datetime.datetime.now().isoformat(timespec='seconds')} ---\n")

    def write(self, data: str) -> int:
        self._orig.write(data)
        self._log.write(data)
        return len(data)

    def flush(self) -> None:
        self._orig.flush()
        self._log.flush()

    def fileno(self) -> int:
        return self._orig.fileno()
from voice_transcribe.agents import diagram as diagram_agent
from voice_transcribe.audio import record_once
from voice_transcribe.llm import run_llm_prompt
from voice_transcribe.loops import conversational
from voice_transcribe.storage import process_and_save
from voice_transcribe.stt import load_transcriber


def parse_args() -> argparse.Namespace:
    env_backend = os.getenv("VOICE_LLM_BACKEND", DEFAULT_LLM_BACKEND).lower()
    if env_backend not in {"claude", "ollama"}:
        env_backend = DEFAULT_LLM_BACKEND
    env_correction_backend = os.getenv("VOICE_CORRECTION_BACKEND", env_backend).lower()
    if env_correction_backend not in {"claude", "ollama"}:
        env_correction_backend = env_backend
    env_process_backend = os.getenv("VOICE_PROCESS_BACKEND", env_backend).lower()
    if env_process_backend not in {"claude", "ollama"}:
        env_process_backend = env_backend

    env_ollama_model = os.getenv("VOICE_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    env_correction_model = os.getenv("VOICE_CORRECTION_OLLAMA_MODEL", env_ollama_model)
    env_process_model = os.getenv("VOICE_PROCESS_OLLAMA_MODEL", env_ollama_model)

    _DEFAULT_PRESET = (
        "--stt-backend faster-whisper --whisper-model medium.en "
        "--correction-backend ollama --process-backend ollama "
        "--correction-ollama-model qwen2.5:7b-instruct "
        "--process-ollama-model qwen2.5:7b-instruct"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--default",
        action="store_true",
        default=False,
        help=(
            "Apply the built-in default preset, ignoring environment variables: "
            + _DEFAULT_PRESET
        ),
    )
    parser.add_argument(
        "--stt-backend",
        choices=["faster-whisper", "mlx-whisper", "granite-speech"],
        default=os.getenv("VOICE_STT_BACKEND", "faster-whisper"),
        help="Speech-to-text backend.",
    )
    parser.add_argument(
        "--whisper-model",
        default=os.getenv("VOICE_WHISPER_MODEL", DEFAULT_MODEL),
        help="faster-whisper model size to use when --stt-backend=faster-whisper.",
    )
    parser.add_argument(
        "--mlx-model",
        default=os.getenv("VOICE_MLX_MODEL", DEFAULT_MLX_MODEL),
        help="mlx-whisper model id to use when --stt-backend=mlx-whisper.",
    )
    parser.add_argument(
        "--granite-model",
        default=os.getenv("VOICE_GRANITE_MODEL", DEFAULT_GRANITE_MODEL),
        help="Granite Speech model id to use when --stt-backend=granite-speech.",
    )
    parser.add_argument(
        "--correction-backend",
        choices=["claude", "ollama"],
        default=env_correction_backend,
        help="LLM backend for correction.",
    )
    parser.add_argument(
        "--process-backend",
        choices=["claude", "ollama"],
        default=env_process_backend,
        help="LLM backend for final processing/saving.",
    )
    parser.add_argument(
        "--correction-ollama-model",
        default=env_correction_model,
        help="Ollama model to use when --correction-backend=ollama.",
    )
    parser.add_argument(
        "--process-ollama-model",
        default=env_process_model,
        help="Ollama model to use when --process-backend=ollama.",
    )
    args = parser.parse_args()

    if args.default:
        args.stt_backend = "faster-whisper"
        args.whisper_model = DEFAULT_MODEL
        args.correction_backend = "ollama"
        args.process_backend = "ollama"
        args.correction_ollama_model = DEFAULT_OLLAMA_MODEL
        args.process_ollama_model = DEFAULT_OLLAMA_MODEL

    return args


def _print_config(args: argparse.Namespace) -> None:
    print("Effective configuration:")
    print(f"  --stt-backend               {args.stt_backend}")
    print(f"  --whisper-model             {args.whisper_model}")
    print(f"  --mlx-model                 {args.mlx_model}")
    print(f"  --granite-model             {args.granite_model}")
    print(f"  --correction-backend        {args.correction_backend}")
    print(f"  --correction-ollama-model   {args.correction_ollama_model}")
    print(f"  --process-backend           {args.process_backend}")
    print(f"  --process-ollama-model      {args.process_ollama_model}")
    print()
    print("Use --default to apply the built-in preset (ignores env vars):")
    print("  --stt-backend faster-whisper --whisper-model medium.en")
    print("  --correction-backend ollama  --correction-ollama-model qwen2.5:7b-instruct")
    print("  --process-backend ollama     --process-ollama-model qwen2.5:7b-instruct")
    print()
    print("Run 'voice --help' for all options.")


_DIAGRAM_SUGGESTION_PROMPT = """\
Read the markdown below and suggest diagrams that would make this document clearer.

Output a JSON array (and nothing else) of objects with keys "label" and "description".
- "label": short title (3-6 words)
- "description": one sentence describing what the diagram should show

If no diagrams would add value, output an empty array: []

Example:
[
  {{"label": "Deployment Sequence", "description": "Steps from code push to production deployment"}},
  {{"label": "Component Overview", "description": "How the API, worker, and database relate to each other"}}
]

Markdown:
{md_text}
"""

_JSON_ARRAY_RE = re.compile(r"\[.*?\]", re.DOTALL)


def _suggest_diagrams(md_text: str, llm_backend: str, ollama_model: str) -> list[dict]:
    prompt = _DIAGRAM_SUGGESTION_PROMPT.format(md_text=md_text)
    raw = run_llm_prompt(prompt, llm_backend, ollama_model)
    m = _JSON_ARRAY_RE.search(raw)
    if not m:
        return []
    try:
        suggestions = json.loads(m.group(0))
        return [s for s in suggestions if isinstance(s, dict) and "label" in s and "description" in s]
    except json.JSONDecodeError:
        return []


_D2_REFINE_PROMPT = """\
You are editing a D2 diagram. Here is the current source:

```d2
{syntax}
```

User feedback: {feedback}

Output the modified D2 syntax inside a d2 fenced code block. No explanation.
"""

_D2_FENCE_RE = re.compile(r"```(?:d2)?\s*(.*?)```", re.DOTALL)


def _label_to_slug(label: str) -> str:
    slug = "".join(c if c.isalnum() else "_" for c in label.lower())
    return "_".join(w for w in slug.split("_") if w)


def _make_session_logger(log_path: pathlib.Path):
    """Return a callable that appends a timestamped line to log_path."""
    def _log(msg: str) -> None:
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        with open(log_path, "a") as f:
            f.write(f"{ts}  {msg}\n")
    return _log


def _refine_diagram_loop(
    output_path: pathlib.Path,
    initial_syntax: str,
    llm_backend: str,
    ollama_model: str,
    transcribe_chunk_fn=None,
    log=None,
) -> None:
    from voice_transcribe.tools.d2 import render as d2_render

    current_syntax = initial_syntax

    while True:
        subprocess.run(["open", str(output_path)])
        try:
            if transcribe_chunk_fn is not None:
                feedback = record_once(transcribe_chunk_fn)
            else:
                feedback = input("  Looks good? [Enter to keep / describe a change] ").strip() or None
        except KeyboardInterrupt:
            print("\n  (refinement cancelled)", flush=True)
            break
        if feedback is None:
            break

        if log:
            log(f"refine feedback: {feedback}")

        prompt = _D2_REFINE_PROMPT.format(syntax=current_syntax, feedback=feedback)
        raw = run_llm_prompt(prompt, llm_backend, ollama_model)

        m = _D2_FENCE_RE.search(raw)
        new_syntax = m.group(1).strip() if m else raw.strip()

        if log:
            log(f"refine syntax ({len(new_syntax)} chars):\n{new_syntax}")

        result = d2_render(new_syntax, output_path)
        if result.get("success"):
            current_syntax = new_syntax
            msg = "  Updated."
            print(msg, flush=True)
            if log:
                log(msg)
        else:
            err = result.get("error", "unknown error")
            print(f"  Render failed: {err}", flush=True)
            print("  (current version kept — try a different description)", flush=True)
            if log:
                log(f"refine render failed: {err}")


def _run_diagram_flow(session_dir: pathlib.Path, llm_backend: str, ollama_model: str, transcribe_chunk_fn=None) -> None:
    md_files = list(session_dir.glob("*.md"))
    if not md_files:
        return
    md_path = md_files[0]
    md_text = md_path.read_text()

    log_path = session_dir / "session.log"
    log = _make_session_logger(log_path)
    log(f"diagram flow start — backend={llm_backend} model={ollama_model}")
    print(f"\n  session log → {log_path}", flush=True)

    print("\nAnalyzing for diagram suggestions...", flush=True)
    log("requesting diagram suggestions from LLM")
    suggestions = _suggest_diagrams(md_text, llm_backend, ollama_model)
    log(f"suggestions received: {[s.get('label') for s in suggestions]}")

    if not suggestions:
        print("  (no diagram suggestions)", flush=True)
        return

    for s in suggestions:
        label = s["label"]
        description = s["description"]
        answer = input(f'\nSuggested: "{label}" — {description}\n  Add it? [y/N] ').strip().lower()
        log(f'suggestion "{label}": user answered "{answer}"')
        if answer != "y":
            continue

        slug = _label_to_slug(label)
        output_path = session_dir / f"{slug}.png"
        print(f"  Rendering {slug}.png ...", flush=True)
        log(f"starting diagram agent for: {description}")

        result = diagram_agent.run(description, output_path, llm_backend, ollama_model, log=log)
        if result.get("success"):
            print(f"  Saved → {output_path}", flush=True)
            log(f"diagram saved: {output_path}")
            with open(md_path, "a") as f:
                f.write(f"\n\n![{label}]({output_path.name})\n")
            _refine_diagram_loop(output_path, result.get("syntax", ""), llm_backend, ollama_model, transcribe_chunk_fn=transcribe_chunk_fn, log=log)
        else:
            err = result.get("error", "unknown error")
            print(f"  Diagram failed: {err}", flush=True)
            log(f"diagram failed: {err}")


def _display_markdown(session_dir: pathlib.Path) -> None:
    try:
        from rich.console import Console
        from rich.markdown import Markdown
    except ImportError:
        return

    md_files = list(session_dir.glob("*.md"))
    if not md_files:
        return
    md_text = md_files[0].read_text()

    # Replace image lines with placeholders (terminal can't render pixels)
    def _img_placeholder(m: re.Match) -> str:
        return f"[image: {m.group(1)}]"

    display_text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", _img_placeholder, md_text)

    console = Console()
    console.print()
    console.rule("Transcript")
    console.print(Markdown(display_text))
    console.rule()


def main() -> None:
    args = parse_args()

    if len(sys.argv) == 1:
        _print_config(args)
        sys.exit(0)

    sys.stderr = _StderrTee(LOG_FILE)
    print(f"  log → {LOG_FILE}", flush=True)

    transcribe_chunk_fn = load_transcriber(
        stt_backend=args.stt_backend,
        faster_model_size=args.whisper_model,
        mlx_model_id=args.mlx_model,
        granite_model_id=args.granite_model,
    )
    clear_screen()
    print(f"STT backend: {args.stt_backend}", flush=True)
    if args.stt_backend == "faster-whisper":
        print(f"Whisper model: {args.whisper_model}", flush=True)
    elif args.stt_backend == "mlx-whisper":
        print(f"MLX model: {args.mlx_model}", flush=True)
    elif args.stt_backend == "granite-speech":
        print(f"Granite model: {args.granite_model}", flush=True)
    print(f"Correction backend: {args.correction_backend}", flush=True)
    if args.correction_backend == "ollama":
        print(f"Correction model: {args.correction_ollama_model}", flush=True)
    print(f"Process backend: {args.process_backend}", flush=True)
    if args.process_backend == "ollama":
        print(f"Process model: {args.process_ollama_model}", flush=True)

    text = conversational.run(
        transcribe_chunk_fn,
        args.correction_backend,
        args.correction_ollama_model,
    )
    if text:
        session_dir = process_and_save(text, args.process_backend, args.process_ollama_model)
        try:
            _run_diagram_flow(session_dir, args.process_backend, args.process_ollama_model, transcribe_chunk_fn=transcribe_chunk_fn)
        except KeyboardInterrupt:
            print("\n  (diagram generation cancelled)", flush=True)
        _display_markdown(session_dir)
    sys.exit(0)
