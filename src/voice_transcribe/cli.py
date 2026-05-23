import argparse
import datetime
import os
import pathlib
import sys

from voice_transcribe.config import (
    DEFAULT_LLM_BACKEND,
    DEFAULT_OLLAMA_MODEL,
    LOG_FILE,
)
from voice_transcribe.deepgram_stream import DEFAULT_DEEPGRAM_MODEL


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


def _get_api_key() -> str:
    key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    if not key:
        print(
            "Error: DEEPGRAM_API_KEY environment variable is not set.\n"
            "Set it with: export DEEPGRAM_API_KEY=your_key_here",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


def parse_args() -> argparse.Namespace:
    env_backend = os.getenv("VOICE_LLM_BACKEND", DEFAULT_LLM_BACKEND).lower()
    if env_backend not in {"claude", "ollama"}:
        env_backend = DEFAULT_LLM_BACKEND

    env_ollama_model = os.getenv("VOICE_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    env_deepgram_model = os.getenv("VOICE_DEEPGRAM_MODEL", DEFAULT_DEEPGRAM_MODEL)

    parser = argparse.ArgumentParser(
        description="Voice transcription and conversation tool (Deepgram STT)."
    )
    parser.add_argument(
        "--deepgram-model",
        default=env_deepgram_model,
        help=f"Deepgram model to use for STT (default: {DEFAULT_DEEPGRAM_MODEL}).",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["claude", "ollama"],
        default=env_backend,
        help="LLM backend for orchestration and processing.",
    )
    parser.add_argument(
        "--ollama-model",
        default=env_ollama_model,
        help="Ollama model to use when --llm-backend=ollama.",
    )
    parser.add_argument(
        "--debug-protocol",
        action="store_true",
        default=False,
        help="Print raw LLM protocol JSON to stderr for debugging.",
    )
    return parser.parse_args()


def _print_config(args: argparse.Namespace) -> None:
    print("Effective configuration:")
    print(f"  --deepgram-model   {args.deepgram_model}")
    print(f"  --llm-backend      {args.llm_backend}")
    if args.llm_backend == "ollama":
        print(f"  --ollama-model     {args.ollama_model}")
    print()
    print("Environment variables:")
    print("  DEEPGRAM_API_KEY   (required)")
    print("  VOICE_DEEPGRAM_MODEL")
    print("  VOICE_LLM_BACKEND")
    print("  VOICE_OLLAMA_MODEL")
    print()
    print("Run 'voice --help' for all options.")


def main() -> None:
    args = parse_args()

    if len(sys.argv) == 1:
        _print_config(args)
        sys.exit(0)

    api_key = _get_api_key()

    sys.stderr = _StderrTee(LOG_FILE)
    print(f"  log → {LOG_FILE}", flush=True)

    from voice_transcribe.audio import clear_screen
    from voice_transcribe.loops.conversation_session import run as run_session

    clear_screen()
    print(f"STT: Deepgram {args.deepgram_model}", flush=True)
    print(f"LLM: {args.llm_backend}", flush=True)
    if args.llm_backend == "ollama":
        print(f"Ollama model: {args.ollama_model}", flush=True)

    run_session(
        api_key=api_key,
        deepgram_model=args.deepgram_model,
        llm_backend=args.llm_backend,
        ollama_model=args.ollama_model,
    )
    sys.exit(0)
