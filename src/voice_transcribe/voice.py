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

import difflib
import os
import argparse
import pathlib
import queue
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import termios
import threading
import time
import tty
import wave
from collections.abc import Callable

import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000   # Hz — Whisper is trained on 16 kHz audio
CHANNELS = 1
DTYPE = "float32"
CHUNK_FRAMES = 512    # frames per read call (~32 ms at 16 kHz)
LIVE_CHUNK_SECONDS = 5  # how often to send audio to the transcription worker

# Model sizes: tiny | base | small | medium | large
DEFAULT_MODEL = "medium.en"
DEFAULT_MLX_MODEL = "mlx-community/whisper-small.en-mlx"

TRANSCRIPT_DIR = pathlib.Path.home() / "transcript"
IDLE_TIMEOUT = 300  # seconds of inactivity before auto-save and exit (5 minutes)

PROCESS_PROMPT_FILE = pathlib.Path(__file__).parent / "process_prompt.md"
DEFAULT_LLM_BACKEND = "claude"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"

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


def write_wav(path: pathlib.Path, audio: np.ndarray) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def load_transcriber(
    stt_backend: str, faster_model_size: str = DEFAULT_MODEL, mlx_model_id: str = DEFAULT_MLX_MODEL
) -> Callable[[np.ndarray], list[str]]:
    if stt_backend == "mlx-whisper":
        try:
            import mlx_whisper
        except ImportError as exc:
            raise RuntimeError(
                "mlx-whisper backend selected but package is not installed. "
                "Install it with: uv pip install mlx-whisper"
            ) from exc

        mps_available = torch.backends.mps.is_available()
        print(
            f"Loading mlx-whisper model '{mlx_model_id}' on {'mps' if mps_available else 'cpu'} "
            "(first run may download weights) ...",
            flush=True,
        )

        def transcribe_chunk(audio_chunk: np.ndarray) -> list[str]:
            tmp_wav = pathlib.Path(tempfile.mktemp(suffix=".wav"))
            write_wav(tmp_wav, audio_chunk)
            try:
                result = mlx_whisper.transcribe(str(tmp_wav), path_or_hf_repo=mlx_model_id)
            finally:
                tmp_wav.unlink(missing_ok=True)
            text = result.get("text", "").strip() if isinstance(result, dict) else str(result).strip()
            return [text] if text else []

        print("mlx-whisper transcriber ready.\n", flush=True)
        return transcribe_chunk

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(
        f"Loading faster-whisper '{faster_model_size}' model on {device} "
        "(first run will download the weights) ...",
        flush=True,
    )
    model = WhisperModel(faster_model_size, device=device, compute_type=compute_type)
    print("faster-whisper model ready.\n", flush=True)

    def transcribe_chunk(audio_chunk: np.ndarray) -> list[str]:
        segs, _ = model.transcribe(
            audio_chunk,
            vad_filter=True,
            vad_parameters={"threshold": 0.5},
            no_speech_threshold=0.6,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            condition_on_previous_text=False,
        )
        texts = []
        for seg in segs:
            text = seg.text.strip()
            if text:
                texts.append(text)
        return texts

    return transcribe_chunk


# ---------------------------------------------------------------------------
# Recording + live transcription
# ---------------------------------------------------------------------------

_RED        = "\033[31m"
_STRIKE_RED = "\033[31m\033[9m"
_GREEN      = "\033[32m"
_RESET      = "\033[0m"
_CLEAR      = "\r" + " " * 40 + "\r"


def clear_screen() -> None:
    print("\033[2J\033[H", end="", flush=True)


def highlight_diff(before: str, after: str) -> str:
    """Return after text with removed words in red strikethrough and added words in green."""
    before_words = before.split()
    after_words = after.split()
    matcher = difflib.SequenceMatcher(None, before_words, after_words)
    result = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            result.append(" ".join(after_words[j1:j2]))
        elif op == "replace":
            result.append(f"{_STRIKE_RED}{' '.join(before_words[i1:i2])}{_RESET}")
            result.append(f"{_GREEN}{' '.join(after_words[j1:j2])}{_RESET}")
        elif op == "delete":
            result.append(f"{_STRIKE_RED}{' '.join(before_words[i1:i2])}{_RESET}")
        elif op == "insert":
            result.append(f"{_GREEN}{' '.join(after_words[j1:j2])}{_RESET}")
    return " ".join(result)


def record_and_transcribe_live(transcribe_chunk_fn: Callable[[np.ndarray], list[str]]) -> tuple[str | None, np.ndarray]:
    """Record audio with live transcription displayed as you speak.

    Press SPACE to start, press any key to stop.
    Words appear on screen as each chunk is transcribed in the background.
    Returns (transcript, full_audio), or (None, empty) if idle timeout is reached.
    """
    all_frames: list[np.ndarray] = []
    start_event = threading.Event()
    stop_event = threading.Event()

    def _key_listener() -> None:
        """Read keypresses via raw terminal stdin. No pynput needed."""
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            # Phase 1: wait for SPACE to start
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0.5)
                if stop_event.is_set():
                    return  # idle timeout fired externally
                if ready:
                    ch = sys.stdin.buffer.read(1)
                    if ch == b'\x03':  # Ctrl+C
                        os.kill(os.getpid(), signal.SIGINT)
                        return
                    if ch == b' ':
                        start_event.set()
                        break
            # Phase 2: wait for any key to stop
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0.5)
                if stop_event.is_set():
                    return
                if ready:
                    ch = sys.stdin.buffer.read(1)
                    if ch == b'\x03':  # Ctrl+C
                        os.kill(os.getpid(), signal.SIGINT)
                        return
                    print("\n[debug] stop key detected; signaling stop_event.", flush=True)
                    stop_event.set()
                    return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    key_thread = threading.Thread(target=_key_listener, daemon=True)
    key_thread.start()

    print("Press SPACE to start recording, press any key to stop. Ctrl+C to quit.")
    if not start_event.wait(timeout=IDLE_TIMEOUT):
        stop_event.set()  # signal key_thread to exit
        key_thread.join(timeout=1.0)
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
            rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
            if rms < 0.005:
                tx_queue.task_done()
                continue
            for text in transcribe_chunk_fn(audio_chunk):
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

    # Callback runs in sounddevice's thread — keeps main thread free for signals
    def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        nonlocal pending, pending_samples
        chunk = indata.copy()
        all_frames.append(chunk)
        pending.append(chunk)
        pending_samples += len(chunk)
        if pending_samples >= CHUNK_SAMPLES:
            tx_queue.put(np.concatenate(pending).flatten())
            pending.clear()
            pending_samples = 0

    t0_record = time.perf_counter()

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
        blocksize=CHUNK_FRAMES, callback=audio_callback,
    ):
        while not stop_event.is_set():
            time.sleep(0.05)  # main thread stays free — Ctrl+C and key events work

    record_dur = time.perf_counter() - t0_record

    # Flush any audio that didn't fill a full chunk
    if pending:
        print(
            f"\n[debug] final flush enqueue: {pending_samples} pending samples queued for transcription.",
            flush=True,
        )
        tx_queue.put(np.concatenate(pending).flatten())

    # Signal worker to stop and wait for remaining transcription to finish
    tx_queue.put(None)
    tx_thread.join()
    key_thread.join(timeout=1.0)

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
# LLM-powered correction and processing
# ---------------------------------------------------------------------------

def run_ollama_prompt(prompt: str, model: str) -> str:
    """Run an Ollama prompt and return stdout."""
    result = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def run_claude_prompt(prompt: str) -> str:
    """Run a Claude prompt and return stdout."""
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True,
        text=True,
        check=True,
        stderr=subprocess.DEVNULL,
    )
    return result.stdout.strip()


def run_llm_prompt(prompt: str, llm_backend: str, ollama_model: str) -> str:
    """Run prompt with selected backend and return stdout."""
    if llm_backend == "ollama":
        return run_ollama_prompt(prompt, ollama_model)
    return run_claude_prompt(prompt)


def correct_with_llm(
    transcript: str, instructions: str, llm_backend: str, ollama_model: str
) -> str:
    """Apply spoken correction instructions to the transcript using the selected LLM backend."""
    if llm_backend == "ollama":
        prompt = (
            "You are editing a transcript.\n"
            "Apply the user's correction instructions to the transcript text.\n"
            "Return only the corrected transcript text, with no commentary.\n\n"
            f"Correction instructions:\n{instructions}\n\n"
            f"Transcript:\n{transcript}\n"
        )
        return run_ollama_prompt(prompt, ollama_model)

    # Default: claude CLI path
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
        stderr=subprocess.DEVNULL,
    )
    corrected = tmp.read_text().strip()
    tmp.unlink()
    return corrected


def process_and_save(transcript: str, llm_backend: str, ollama_model: str) -> None:
    """Process the final transcript with the selected LLM backend and save to ~/transcript/."""
    TRANSCRIPT_DIR.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tmp = pathlib.Path(f".rec.tmp.{timestamp}.md")
    print(f"\n  (working file: {tmp})")

    print(f"\r{_RED}●{_RESET} PROCESSING...", end="", flush=True)

    # Step 1: build structured markdown using shared prompt file
    process_prompt = PROCESS_PROMPT_FILE.read_text().replace("{transcript}", transcript)
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


# ---------------------------------------------------------------------------
# Review prompt
# ---------------------------------------------------------------------------

def prompt_review(transcript: str) -> tuple[str, str]:
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


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stt-backend",
        choices=["faster-whisper", "mlx-whisper"],
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
    return parser.parse_args()


def main() -> None:
    # sys.stderr = open(os.devnull, "w")  # uncomment once stable
    args = parse_args()

    transcribe_chunk_fn = load_transcriber(
        stt_backend=args.stt_backend,
        faster_model_size=args.whisper_model,
        mlx_model_id=args.mlx_model,
    )
    clear_screen()
    print(f"STT backend: {args.stt_backend}", flush=True)
    if args.stt_backend == "faster-whisper":
        print(f"Whisper model: {args.whisper_model}", flush=True)
    else:
        print(f"MLX model: {args.mlx_model}", flush=True)
    print(f"Correction backend: {args.correction_backend}", flush=True)
    if args.correction_backend == "ollama":
        print(f"Correction model: {args.correction_ollama_model}", flush=True)
    print(f"Process backend: {args.process_backend}", flush=True)
    if args.process_backend == "ollama":
        print(f"Process model: {args.process_ollama_model}", flush=True)

    tmp_file = pathlib.Path(tempfile.mktemp(suffix=".txt"))

    def append_chunk(text: str) -> None:
        with open(tmp_file, "a") as f:
            f.write(text + "\n\n")

    def show_accumulated() -> None:
        if tmp_file.exists() and tmp_file.stat().st_size > 0:
            print(tmp_file.read_text())
            print("-" * 40)

    def finalize() -> None:
        if tmp_file.exists() and tmp_file.stat().st_size > 0:
            process_and_save(
                tmp_file.read_text(),
                args.process_backend,
                args.process_ollama_model,
            )
            tmp_file.unlink(missing_ok=True)

    try:
        while True:
            clear_screen()
            show_accumulated()
            print()

            transcript, _ = record_and_transcribe_live(transcribe_chunk_fn)

            if transcript is None:
                print("\nIdle timeout.")
                finalize()
                sys.exit(0)

            if not transcript:
                print("No speech detected — try again.\n")
                continue

            current = transcript
            display = transcript

            while True:
                clear_screen()
                show_accumulated()
                print(f"Current:\n\n  {display}\n")

                action, _ = prompt_review(current)

                if action == "timeout":
                    append_chunk(current)
                    print("\nIdle timeout — saving and exiting.")
                    finalize()
                    sys.exit(0)

                if action == "exit":
                    append_chunk(current)
                    finalize()
                    print("\nGoodbye.")
                    sys.exit(0)

                if action == "add":
                    append_chunk(current)
                    break  # outer loop: show accumulated → record next chunk

                if action == "correct":
                    print()
                    instructions, _ = record_and_transcribe_live(transcribe_chunk_fn)
                    if not instructions:
                        print("No instructions heard — try again.\n")
                        continue
                    _no_change_phrases = ("do nothing", "make no changes", "no changes", "never mind", "cancel")
                    if any(p in instructions.lower() for p in _no_change_phrases):
                        continue
                    print(f"\r{_RED}●{_RESET} CORRECTING...", end="", flush=True)
                    corrected = correct_with_llm(
                        current,
                        instructions,
                        args.correction_backend,
                        args.correction_ollama_model,
                    )
                    display = highlight_diff(current, corrected)
                    current = corrected
                    print(_CLEAR, end="", flush=True)

    except KeyboardInterrupt:
        if tmp_file.exists() and tmp_file.stat().st_size > 0:
            try:
                finalize()
            except Exception:
                pass
        print("\nGoodbye.")
        sys.exit(0)


if __name__ == "__main__":
    main()
