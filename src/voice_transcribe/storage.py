import pathlib
import shutil
import time

from voice_transcribe import vector
from voice_transcribe.audio import _CLEAR, _RED, _RESET
from voice_transcribe.config import TRANSCRIPT_DIR
from voice_transcribe.llm import run_llm_prompt
from voice_transcribe.prompts import load_process_prompt


def process_and_save(transcript: str, llm_backend: str, ollama_model: str) -> None:
    """Process the final transcript with the selected LLM backend and save to ~/transcript/."""
    TRANSCRIPT_DIR.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tmp = pathlib.Path(f".rec.tmp.{timestamp}.md")
    print(f"\n  (working file: {tmp})")

    print(f"\r{_RED}●{_RESET} PROCESSING...", end="", flush=True)

    # Step 1: build structured markdown using shared prompt file
    process_prompt = load_process_prompt(transcript)
    processed = run_llm_prompt(process_prompt, llm_backend, ollama_model)
    tmp.write_text(processed)

    # Step 2: get a 5-word filename summary from selected backend
    filename_prompt = (
        "Output only a 5-word filename summary for the text below: "
        "lowercase words separated by underscores, no file extension, no punctuation, nothing else.\n\n"
        f"{processed}\n"
    )
    raw = run_llm_prompt(filename_prompt, llm_backend, ollama_model).lower()

    filename = "".join(c if c.isalnum() or c == "_" else "_" for c in raw)
    filename = "_".join(w for w in filename.split("_") if w) + ".md"

    print(_CLEAR, end="", flush=True)

    dest = TRANSCRIPT_DIR / filename
    shutil.move(str(tmp), dest)  # only removes tmp on success
    print(f"\nSaved → {dest}")

    # Phase 2: index the saved document in the vector store
    vector.on_doc_saved(dest)
