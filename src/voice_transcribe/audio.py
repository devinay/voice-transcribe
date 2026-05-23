import difflib
import os
import queue
import select
import signal
import sys
import termios
import threading
import time
import tty
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from voice_transcribe.config import (
    CHUNK_FRAMES,
    CHANNELS,
    DTYPE,
    IDLE_TIMEOUT,
    MIN_STREAM_AUDIO_SECONDS,
    SAMPLE_RATE,
    STREAM_BUFFER_SECONDS,
    STREAM_DECODE_SECONDS,
    STREAM_WINDOW_SECONDS,
)

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


def _norm_token(token: str) -> str:
    return "".join(ch for ch in token.lower() if ch.isalnum())


def _run_transcription_worker(
    transcribe_chunk_fn: Callable[[np.ndarray], list[str]],
    tx_queue: queue.Queue,
    transcript_words: list[str],
) -> threading.Thread:
    """Spin up the background transcription worker thread and return it."""
    total_tx_wall = 0.0
    total_tx_cpu = 0.0

    decode_every_samples = int(STREAM_DECODE_SECONDS * SAMPLE_RATE)
    decode_window_samples = int(STREAM_WINDOW_SECONDS * SAMPLE_RATE)
    max_buffer_samples = int(STREAM_BUFFER_SECONDS * SAMPLE_RATE)
    min_decode_samples = int(MIN_STREAM_AUDIO_SECONDS * SAMPLE_RATE)

    def worker() -> None:
        nonlocal total_tx_wall, total_tx_cpu
        rolling_frames: list[np.ndarray] = []
        rolling_samples = 0
        samples_since_decode = 0
        prev_window_words: list[str] = []

        def _tail_audio(frames: list[np.ndarray], total: int, target: int) -> np.ndarray:
            if not frames:
                return np.array([], dtype=np.float32)
            if total <= target:
                return np.concatenate(frames).flatten()
            need = target
            selected: list[np.ndarray] = []
            for frame in reversed(frames):
                fl = len(frame)
                if fl <= need:
                    selected.append(frame)
                    need -= fl
                else:
                    selected.append(frame[-need:])
                    need = 0
                if need == 0:
                    break
            selected.reverse()
            return np.concatenate(selected).flatten()

        def _decode_once() -> None:
            nonlocal total_tx_wall, total_tx_cpu, prev_window_words
            if rolling_samples < min_decode_samples:
                return
            audio_window = _tail_audio(rolling_frames, rolling_samples, decode_window_samples)
            if len(audio_window) == 0:
                return
            rms = float(np.sqrt(np.mean(audio_window ** 2)))
            if rms < 0.005:
                return
            t0w = time.perf_counter()
            t0c = time.process_time()
            window_text = " ".join(transcribe_chunk_fn(audio_window)).strip()
            total_tx_wall += time.perf_counter() - t0w
            total_tx_cpu += time.process_time() - t0c
            if not window_text:
                return
            curr_words = window_text.split()
            new_words = _new_words_from_overlap(prev_window_words, curr_words)
            if new_words:
                transcript_words.extend(new_words)
                print(" ".join(new_words) + " ", end="", flush=True)
            prev_window_words = curr_words

        while True:
            frame = tx_queue.get()
            if frame is None:
                _decode_once()
                tx_queue.task_done()
                break
            rolling_frames.append(frame)
            frame_len = len(frame)
            rolling_samples += frame_len
            samples_since_decode += frame_len
            while rolling_samples > max_buffer_samples and rolling_frames:
                dropped = rolling_frames.pop(0)
                rolling_samples -= len(dropped)
            if samples_since_decode >= decode_every_samples:
                _decode_once()
                samples_since_decode = 0
            tx_queue.task_done()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t


def record_once(transcribe_chunk_fn: Callable[[np.ndarray], list[str]]) -> str | None:
    """Record a single voice utterance for short prompts (e.g. diagram refinement).

    Press ENTER to skip (keep as-is).
    Press SPACE to start speaking, then any key to stop.
    Returns the transcript string, or None if ENTER was pressed or nothing was spoken.
    """
    start_event = threading.Event()
    stop_event = threading.Event()
    skip_event = threading.Event()

    def _key_listener() -> None:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0.5)
                if stop_event.is_set():
                    return
                if ready:
                    ch = sys.stdin.buffer.read(1)
                    if ch == b'\x03':
                        os.kill(os.getpid(), signal.SIGINT)
                        return
                    if ch in (b'\r', b'\n'):
                        skip_event.set()
                        stop_event.set()
                        return
                    if ch == b' ':
                        start_event.set()
                        break
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0.5)
                if stop_event.is_set():
                    return
                if ready:
                    ch = sys.stdin.buffer.read(1)
                    if ch == b'\x03':
                        os.kill(os.getpid(), signal.SIGINT)
                        return
                    stop_event.set()
                    return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    print("  Press SPACE to describe a change, ENTER to keep.", flush=True)
    key_thread = threading.Thread(target=_key_listener, daemon=True)
    key_thread.start()

    while not start_event.is_set() and not skip_event.is_set():
        time.sleep(0.05)

    if skip_event.is_set():
        key_thread.join(timeout=1.0)
        return None

    print(f"\n  {_RED}●{_RESET} ", end="", flush=True)

    tx_queue: queue.Queue = queue.Queue()
    transcript_words: list[str] = []
    tx_thread = _run_transcription_worker(transcribe_chunk_fn, tx_queue, transcript_words)

    def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        tx_queue.put(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
        blocksize=CHUNK_FRAMES, callback=audio_callback,
    ):
        while not stop_event.is_set():
            time.sleep(0.05)

    tx_queue.put(None)
    tx_thread.join()
    key_thread.join(timeout=1.0)

    transcript = " ".join(transcript_words).strip()
    print(flush=True)
    return transcript or None


def record_and_transcribe_live(transcribe_chunk_fn: Callable[[np.ndarray], list[str]]) -> tuple[str | None, np.ndarray]:
    """Record audio with live transcription displayed as you speak.

    Press SPACE to start, press any key to stop.
    Words appear on screen continuously as audio streams to a background worker.
    Returns (transcript, full_audio), or (None, empty) if idle timeout is reached.
    """
    all_frames: list[np.ndarray] = []
    start_event = threading.Event()
    stop_event = threading.Event()

    def _key_listener() -> None:
        """Read keypresses via cbreak terminal stdin. No pynput needed."""
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

    tx_queue: queue.Queue = queue.Queue()
    transcript_words: list[str] = []
    tx_thread = _run_transcription_worker(transcribe_chunk_fn, tx_queue, transcript_words)

    # Callback runs in sounddevice's thread — keeps main thread free for signals
    def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        chunk = indata.copy()
        all_frames.append(chunk)
        tx_queue.put(chunk)

    t0_record = time.perf_counter()

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
        blocksize=CHUNK_FRAMES, callback=audio_callback,
    ):
        while not stop_event.is_set():
            time.sleep(0.05)  # main thread stays free — Ctrl+C and key events work

    record_dur = time.perf_counter() - t0_record

    # Signal worker to stop and wait for remaining transcription to finish
    tx_queue.put(None)
    tx_thread.join()
    key_thread.join(timeout=1.0)

    full_audio = (
        np.concatenate(all_frames).flatten() if all_frames
        else np.array([], dtype=np.float32)
    )
    transcript = " ".join(transcript_words)

    print(f"\n  [{record_dur:.1f}s recorded]")
    return transcript, full_audio
