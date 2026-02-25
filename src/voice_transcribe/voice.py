#!/usr/bin/env python3
"""Voice recorder with local Whisper transcription.

Usage:
    python -m voice_transcribe.voice          # run as module
    voice                              # if installed via pyproject.toml scripts

Controls:
    Press SPACE           → start recording (words appear live as you speak)
    Press SPACE again     → stop recording
    Ctrl+C               → quit
"""

import pathlib
import queue
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from pynput import keyboard
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000   # Hz — Whisper is trained on 16 kHz audio
CHANNELS = 1
DTYPE = "float32"
CHUNK_FRAMES = 512    # frames per read call (~32 ms at 16 kHz)
LIVE_CHUNK_SECONDS = 3  # how often to send audio to the transcription worker

# Model sizes: tiny | base | small | medium | large
DEFAULT_MODEL = "medium.en"

TRANSCRIPT_DIR = pathlib.Path.home() / "transcript"
IDLE_TIMEOUT = 300  # seconds of inactivity before auto-save and exit (5 minutes)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_summariser():
    model_id = "philschmid/bart-large-cnn-samsum"
    print(
        f"Loading summarisation model '{model_id}' "
        "(first run will download the weights) ...",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    if torch.cuda.is_available():
        model = model.to("cuda")
    print("Summarisation model ready.\n", flush=True)
    return model, tokenizer


def load_model(size: str = DEFAULT_MODEL) -> WhisperModel:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(
        f"Loading Whisper '{size}' model on {device} "
        "(first run will download the weights) ...",
        flush=True,
    )
    model = WhisperModel(size, device=device, compute_type=compute_type)
    print("Model ready.\n", flush=True)
    return model


# ---------------------------------------------------------------------------
# Recording + live transcription
# ---------------------------------------------------------------------------

_RED   = "\033[31m"
_RESET = "\033[0m"
_CLEAR = "\r" + " " * 40 + "\r"


def record_and_transcribe_live(model: WhisperModel) -> tuple[str | None, np.ndarray]:
    """Record audio with live transcription displayed as you speak.

    Press SPACE to start, press any key to stop.
    Words appear on screen as each chunk is transcribed in the background.
    Returns (transcript, full_audio), or (None, empty) if idle timeout is reached.
    """
    all_frames: list[np.ndarray] = []
    start_event = threading.Event()
    stop_event = threading.Event()

    def on_press(key: keyboard.Key) -> bool | None:
        if not start_event.is_set():
            if key == keyboard.Key.space:
                start_event.set()
        else:
            stop_event.set()
            return False  # stops the listener

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print("Press SPACE to start recording, press any key to stop. Ctrl+C to quit.")
    if not start_event.wait(timeout=IDLE_TIMEOUT):
        listener.stop()
        return None, np.array([], dtype=np.float32)
    print(f"\n  {_RED}●{_RESET} ", end="", flush=True)

    # Background worker: transcribes chunks and prints words as they arrive
    tx_queue: queue.Queue = queue.Queue()
    transcript_parts: list[str] = []
    total_tx_wall = 0.0
    total_tx_cpu = 0.0

    def transcription_worker() -> None:
        nonlocal total_tx_wall, total_tx_cpu
        while True:
            audio_chunk = tx_queue.get()
            if audio_chunk is None:
                break
            t0w = time.perf_counter()
            t0c = time.process_time()
            segments, _ = model.transcribe(audio_chunk, vad_filter=True)
            for seg in segments:
                text = seg.text.strip()
                if text:
                    transcript_parts.append(text)
                    print(text + " ", end="", flush=True)
            total_tx_wall += time.perf_counter() - t0w
            total_tx_cpu += time.process_time() - t0c
            tx_queue.task_done()

    tx_thread = threading.Thread(target=transcription_worker, daemon=True)
    tx_thread.start()

    CHUNK_SAMPLES = int(LIVE_CHUNK_SECONDS * SAMPLE_RATE)
    pending: list[np.ndarray] = []
    pending_samples = 0
    t0_record = time.perf_counter()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE) as stream:
        while not stop_event.is_set():
            chunk, _ = stream.read(CHUNK_FRAMES)
            all_frames.append(chunk.copy())
            pending.append(chunk.copy())
            pending_samples += len(chunk)
            if pending_samples >= CHUNK_SAMPLES:
                tx_queue.put(np.concatenate(pending).flatten())
                pending = []
                pending_samples = 0

    record_dur = time.perf_counter() - t0_record

    # Flush any audio that didn't fill a full chunk
    if pending:
        tx_queue.put(np.concatenate(pending).flatten())

    # Signal worker to stop and wait for remaining transcription to finish
    tx_queue.put(None)
    tx_thread.join()
    listener.join()

    full_audio = (
        np.concatenate(all_frames).flatten() if all_frames
        else np.array([], dtype=np.float32)
    )
    transcript = " ".join(transcript_parts)

    print(f"\n  [{record_dur:.1f}s recorded · {total_tx_wall:.1f}s transcription · CPU {total_tx_cpu:.1f}s]")
    return transcript, full_audio


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------

def summarise(summariser, text: str) -> str:
    """Summarise transcript text using the local BART model."""
    if not text:
        return ""
    model, tokenizer = summariser
    words = text.split()
    clipped = " ".join(words[:900])
    inputs = tokenizer(clipped, return_tensors="pt", max_length=1024, truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    ids = model.generate(inputs["input_ids"], max_length=60, min_length=10, num_beams=4)
    return tokenizer.decode(ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Claude-powered correction
# ---------------------------------------------------------------------------

def correct_with_claude(transcript: str, instructions: str) -> str:
    """Apply spoken correction instructions to the transcript using the claude CLI."""
    tmp = pathlib.Path(tempfile.mktemp(suffix=".md"))
    tmp.write_text(transcript)
    subprocess.run(
        [
            "claude", "-p",
            f"Edit the file {tmp} — apply these correction instructions to the transcript "
            f"text. Only the corrected transcript text should remain in the file, no "
            f"commentary: {instructions}",
            "--allowedTools", "Edit,Write,Read",
        ],
        check=True,
    )
    corrected = tmp.read_text().strip()
    tmp.unlink()
    return corrected


def process_and_save(transcript: str) -> None:
    """Process the final transcript with claude and save to ~/transcript/."""
    TRANSCRIPT_DIR.mkdir(exist_ok=True)

    tmp = pathlib.Path(tempfile.mktemp(suffix=".md"))
    tmp.write_text(transcript)

    print(f"\r{_RED}●{_RESET} PROCESSING...", end="", flush=True)

    # Step 1: Have claude write the full structured markdown to the temp file
    subprocess.run(
        [
            "claude", "-p",
            f"Replace the entire contents of {tmp} with a structured markdown document "
            f"based on the transcript currently in the file. Use exactly these sections:\n\n"
            f"# [Title extracted from transcript]\n\n"
            f"## Summary\n[2-3 sentence summary]\n\n"
            f"## Transcript\n[original transcript, unchanged]\n\n"
            f"## Processed Transcript\n[correct errors, apply punctuation, break into "
            f"paragraphs where the topic changes]\n\n"
            f"## Next Actions\n[bullet list of action items; leave blank if purely informational]",
            "--allowedTools", "Edit,Write,Read",
        ],
        check=True,
    )

    # Step 2: Get the 5-word filename from claude (stdout only)
    result = subprocess.run(
        [
            "claude", "-p",
            f"Read the file {tmp} and output only a 5-word filename summary: "
            f"lowercase words separated by underscores, no file extension, no punctuation, "
            f"nothing else.",
            "--allowedTools", "Read",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    raw = result.stdout.strip().lower()
    filename = "".join(c if c.isalnum() or c == "_" else "_" for c in raw)
    filename = "_".join(w for w in filename.split("_") if w) + ".md"

    print(_CLEAR, end="", flush=True)

    dest = TRANSCRIPT_DIR / filename
    shutil.move(str(tmp), dest)
    print(f"\nSaved → {dest}")


# ---------------------------------------------------------------------------
# Review prompt
# ---------------------------------------------------------------------------

def prompt_review(transcript: str) -> tuple[str, str]:
    """Ask the user what to do with the transcript.

    Returns (action, text) where action is 'proceed', 'add', 'correct', 'exit', or 'timeout'.
    """
    while True:
        sys.stdout.write("  [P]roceed  /  [A]dd  /  [C]orrect  /  e[X]it ? ")
        sys.stdout.flush()
        ready, _, _ = select.select([sys.stdin], [], [], IDLE_TIMEOUT)
        if not ready:
            return "timeout", transcript
        try:
            choice = sys.stdin.readline().strip().lower()
        except EOFError:
            return "proceed", transcript

        if choice in ("p", ""):
            return "proceed", transcript
        if choice == "a":
            return "add", transcript
        if choice == "c":
            return "correct", transcript
        if choice == "x":
            return "exit", transcript
        print("  Please type 'p', 'a', 'c', or 'x'.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    summariser = load_summariser()
    model = load_model()

    try:
        while True:
            print()
            transcript, _ = record_and_transcribe_live(model)

            if transcript is None:
                print("\nIdle timeout — no activity detected. Goodbye.")
                sys.exit(0)

            if not transcript:
                print("No speech detected — try again.\n")
                continue

            print(f"\r{_RED}●{_RESET} SUMMARISING...", end="", flush=True)
            summary = summarise(summariser, transcript)
            print(_CLEAR, end="", flush=True)

            while True:
                print(f"\nSummary:\n  {summary}\n")
                print(f"Transcript:\n  {transcript}\n")

                action, _ = prompt_review(transcript)

                if action == "timeout":
                    print("\nIdle timeout — saving and exiting.")
                    process_and_save(transcript)
                    sys.exit(0)

                if action == "proceed":
                    process_and_save(transcript)
                    break

                if action == "exit":
                    process_and_save(transcript)
                    print("\nGoodbye.")
                    sys.exit(0)

                if action == "correct":
                    print()
                    instructions, _ = record_and_transcribe_live(model)
                    if not instructions:
                        print("No instructions heard — try again.\n")
                        continue
                    _no_change_phrases = ("do nothing", "make no changes", "no changes", "never mind", "cancel")
                    if any(p in instructions.lower() for p in _no_change_phrases):
                        continue
                    print(f"\r{_RED}●{_RESET} CORRECTING...", end="", flush=True)
                    transcript = correct_with_claude(transcript, instructions)
                    print(_CLEAR, end="", flush=True)
                    print(f"\r{_RED}●{_RESET} SUMMARISING...", end="", flush=True)
                    summary = summarise(summariser, transcript)
                    print(_CLEAR, end="", flush=True)
                    # Loop back — corrected transcript + summary will be shown at top

                if action == "add":
                    print()
                    extra, _ = record_and_transcribe_live(model)
                    if not extra:
                        print("No speech detected — try again.\n")
                        continue
                    transcript = transcript + " " + extra
                    print(f"\r{_RED}●{_RESET} SUMMARISING...", end="", flush=True)
                    summary = summarise(summariser, transcript)
                    print(_CLEAR, end="", flush=True)

            print("-" * 40)

    except KeyboardInterrupt:
        print("\nGoodbye.")
        sys.exit(0)


if __name__ == "__main__":
    main()
