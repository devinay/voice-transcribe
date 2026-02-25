#!/usr/bin/env python3
"""Voice recorder with local Whisper transcription.

Usage:
    python -m test_repo.voice          # run as module
    voice                              # if installed via pyproject.toml scripts

Controls:
    Press SPACE           → start recording (words appear live as you speak)
    Press SPACE again     → stop recording
    Ctrl+C               → quit
"""

import queue
import sys
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


def record_and_transcribe_live(model: WhisperModel) -> tuple[str, np.ndarray]:
    """Record audio with live transcription displayed as you speak.

    Press SPACE to start, press SPACE again to stop.
    Words appear on screen as each chunk is transcribed in the background.
    Returns (transcript, full_audio).
    """
    all_frames: list[np.ndarray] = []
    start_event = threading.Event()
    stop_event = threading.Event()
    press_count = 0

    def on_press(key: keyboard.Key) -> bool | None:
        nonlocal press_count
        if key == keyboard.Key.space:
            press_count += 1
            if press_count == 1:
                start_event.set()
            elif press_count == 2:
                stop_event.set()
                return False  # stops the listener

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print("Press SPACE to start recording, press SPACE again to stop. Ctrl+C to quit.")
    start_event.wait()
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
# Confirmation / editing
# ---------------------------------------------------------------------------

def prompt_review(transcript: str) -> tuple[str, str]:
    """Ask the user what to do with the transcript.

    Returns (action, text) where action is 'proceed', 'edit', 'add', or 'exit'.
    """
    while True:
        try:
            choice = input("  [P]roceed  /  [E]dit  /  [A]dd  /  e[X]it ? ").strip().lower()
        except EOFError:
            return "proceed", transcript

        if choice in ("p", ""):
            return "proceed", transcript
        if choice == "e":
            try:
                edited = input("  Your edit: ").strip()
            except EOFError:
                return "proceed", transcript
            return "edit", edited
        if choice == "a":
            return "add", transcript
        if choice == "x":
            return "exit", transcript
        print("  Please type 'p', 'e', 'a', or 'x'.")


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

            if not transcript:
                print("No speech detected — try again.\n")
                continue

            print(f"\r{_RED}●{_RESET} SUMMARISING...", end="", flush=True)
            summary = summarise(summariser, transcript)
            print(_CLEAR, end="", flush=True)

            while True:
                print(f"\nSummary:\n  {summary}\n")
                print(f"Transcript:\n  {transcript}\n")

                action, result = prompt_review(transcript)

                if action == "exit":
                    print("\nGoodbye.")
                    sys.exit(0)

                if action in ("proceed", "edit"):
                    final = result
                    break

                # action == "add": record another clip and append
                print()
                extra, _ = record_and_transcribe_live(model)

                if not extra:
                    print("No speech detected — try again.\n")
                    continue

                transcript = transcript + " " + extra

                print(f"\r{_RED}●{_RESET} SUMMARISING...", end="", flush=True)
                summary = summarise(summariser, transcript)
                print(_CLEAR, end="", flush=True)

            print(f"\nFinal text:\n  {final}\n")
            print("-" * 40)

    except KeyboardInterrupt:
        print("\nGoodbye.")
        sys.exit(0)


if __name__ == "__main__":
    main()
