import pathlib
import select
import sys
import tempfile

from voice_transcribe.audio import _CLEAR, _RED, _RESET, clear_screen, highlight_diff, record_and_transcribe_live
from voice_transcribe.config import IDLE_TIMEOUT
from voice_transcribe.llm import correct_with_llm


def _prompt_review(transcript: str) -> tuple[str, str]:
    """Ask the user what to do with the current transcript chunk.

    Returns (action, text) where action is 'add', 'correct', 'exit', or 'timeout'.
    """
    while True:
        sys.stdout.write("  [A]dd  /  [C]orrect  /  e[X]it ? ")
        sys.stdout.flush()
        ready, _, _ = select.select([sys.stdin], [], [], IDLE_TIMEOUT)
        if not ready:
            return "timeout", transcript
        try:
            choice = sys.stdin.readline().strip().lower()
        except EOFError:
            return "exit", transcript

        if choice == "a":
            return "add", transcript
        if choice == "c":
            return "correct", transcript
        if choice == "x":
            return "exit", transcript
        print("  Please type 'a', 'c', or 'x'.")


def run(transcribe_fn, correction_backend: str, correction_ollama_model: str) -> str:
    """Run the record→review→add/correct/exit loop.

    Returns the accumulated transcript text (possibly empty). Caller is
    responsible for processing and saving.
    """
    tmp_file = pathlib.Path(tempfile.mktemp(suffix=".txt"))

    def _append(text: str) -> None:
        with open(tmp_file, "a") as f:
            f.write(text + "\n\n")

    def _show() -> None:
        if tmp_file.exists() and tmp_file.stat().st_size > 0:
            print(tmp_file.read_text())
            print("-" * 40)

    def _drain() -> str:
        if tmp_file.exists():
            text = tmp_file.read_text().strip()
            tmp_file.unlink(missing_ok=True)
            return text
        return ""

    try:
        while True:
            clear_screen()
            _show()
            print()

            transcript, _ = record_and_transcribe_live(transcribe_fn)

            if transcript is None:
                print("\nIdle timeout.")
                return _drain()

            if not transcript:
                print("No speech detected — try again.\n")
                continue

            current = transcript
            display = transcript

            while True:
                clear_screen()
                _show()
                print(f"Current:\n\n  {display}\n")

                action, _ = _prompt_review(current)

                if action == "timeout":
                    _append(current)
                    print("\nIdle timeout — saving and exiting.")
                    return _drain()

                if action == "exit":
                    _append(current)
                    print("\nGoodbye.")
                    return _drain()

                if action == "add":
                    _append(current)
                    break

                if action == "correct":
                    print()
                    instructions, _ = record_and_transcribe_live(transcribe_fn)
                    if not instructions:
                        print("No instructions heard — try again.\n")
                        continue
                    _no_change_phrases = (
                        "do nothing",
                        "make no changes",
                        "no changes",
                        "never mind",
                        "cancel",
                    )
                    if any(p in instructions.lower() for p in _no_change_phrases):
                        continue
                    print(f"\r{_RED}●{_RESET} CORRECTING...", end="", flush=True)
                    corrected = correct_with_llm(
                        current,
                        instructions,
                        correction_backend,
                        correction_ollama_model,
                    )
                    display = highlight_diff(current, corrected)
                    current = corrected
                    print(_CLEAR, end="", flush=True)

    except KeyboardInterrupt:
        print("\nGoodbye.")
        return _drain()
