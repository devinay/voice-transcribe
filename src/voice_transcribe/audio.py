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

    # Background worker: consumes audio frames and performs streaming decode
    tx_queue: queue.Queue = queue.Queue()
    transcript_words: list[str] = []
    total_tx_wall = 0.0
    total_tx_cpu = 0.0

    def _tail_audio(frames: list[np.ndarray], total_samples: int, target_samples: int) -> np.ndarray:
        if not frames:
            return np.array([], dtype=np.float32)
        if total_samples <= target_samples:
            return np.concatenate(frames).flatten()
        need = target_samples
        selected: list[np.ndarray] = []
        for frame in reversed(frames):
            frame_len = len(frame)
            if frame_len <= need:
                selected.append(frame)
                need -= frame_len
            else:
                selected.append(frame[-need:])
                need = 0
            if need == 0:
                break
        selected.reverse()
        return np.concatenate(selected).flatten()

    def _new_words_from_overlap(prev_words: list[str], curr_words: list[str]) -> list[str]:
        if not prev_words:
            return curr_words
        prev_norm = [_norm_token(w) for w in prev_words]
        curr_norm = [_norm_token(w) for w in curr_words]

        max_overlap = min(len(prev_norm), len(curr_norm))
        for k in range(max_overlap, 0, -1):
            if prev_norm[-k:] == curr_norm[:k]:
                return curr_words[k:]

        # Fuzzy fallback: anchor on the latest matching block and append only new tail words.
        matcher = difflib.SequenceMatcher(None, prev_norm, curr_norm, autojunk=False)
        blocks = [b for b in matcher.get_matching_blocks() if b.size > 0]
        if not blocks:
            return []
        block = max(blocks, key=lambda b: (b.a + b.size, b.size))
        end_in_curr = block.b + block.size
        if end_in_curr >= len(curr_words):
            return []
        return curr_words[end_in_curr:]

    def transcription_worker() -> None:
        nonlocal total_tx_wall, total_tx_cpu
        rolling_frames: list[np.ndarray] = []
        rolling_samples = 0
        samples_since_decode = 0
        prev_window_words: list[str] = []
        decode_every_samples = int(STREAM_DECODE_SECONDS * SAMPLE_RATE)
        decode_window_samples = int(STREAM_WINDOW_SECONDS * SAMPLE_RATE)
        max_buffer_samples = int(STREAM_BUFFER_SECONDS * SAMPLE_RATE)
        min_decode_samples = int(MIN_STREAM_AUDIO_SECONDS * SAMPLE_RATE)

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
                _decode_once()  # final decode before exiting
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

    tx_thread = threading.Thread(target=transcription_worker, daemon=True)
    tx_thread.start()

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

    print(f"\n  [{record_dur:.1f}s recorded · {total_tx_wall:.1f}s transcription · CPU {total_tx_cpu:.1f}s]")
    return transcript, full_audio
