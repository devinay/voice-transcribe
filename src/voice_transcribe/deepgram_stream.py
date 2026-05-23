"""Deepgram streaming client for push-to-talk conversation sessions.

Session boundaries are controlled by the user (SPACE to start, any key to stop).
Deepgram VAD/utterance-end detection is intentionally not used.

Public entry point:
    run_push_to_talk_session(api_key, model, on_partial) -> str
"""
from __future__ import annotations

import os
import queue
import select
import signal
import sys
import termios
import threading
import tty
from collections.abc import Callable

import numpy as np
import sounddevice as sd
from deepgram import DeepgramClient
from deepgram.listen.v2.types.listen_v2turn_info import ListenV2TurnInfo

from voice_transcribe.config import CHANNELS, CHUNK_FRAMES, DTYPE, SAMPLE_RATE

DEFAULT_DEEPGRAM_MODEL = "flux-general-en"

_RED = "\033[31m"
_RESET = "\033[0m"
_CLEAR = "\r" + " " * 80 + "\r"


def run_push_to_talk_session(
    api_key: str,
    model: str = DEFAULT_DEEPGRAM_MODEL,
    on_partial: Callable[[str], None] | None = None,
) -> str:
    """Run one push-to-talk session.

    Waits for SPACE to start, any key to stop.
    Streams mic audio to Deepgram, calls on_partial with live transcript updates.
    Returns the final transcript string, or empty string if nothing was spoken.
    """
    start_event = threading.Event()
    stop_event = threading.Event()
    audio_queue: queue.Queue[bytes | None] = queue.Queue()

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
                    if ch == b"\x03":
                        os.kill(os.getpid(), signal.SIGINT)
                        return
                    if ch == b" ":
                        start_event.set()
                        break
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0.5)
                if stop_event.is_set():
                    return
                if ready:
                    ch = sys.stdin.buffer.read(1)
                    if ch == b"\x03":
                        os.kill(os.getpid(), signal.SIGINT)
                        return
                    stop_event.set()
                    return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    print("Press SPACE to start recording, press any key to stop. Ctrl+C to quit.")
    key_thread = threading.Thread(target=_key_listener, daemon=True)
    key_thread.start()

    start_event.wait()
    print(f"\n  {_RED}●{_RESET} ", end="", flush=True)

    def _audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if not stop_event.is_set():
            pcm = (indata * 32767).astype(np.int16).tobytes()
            audio_queue.put(pcm)

    final_transcript = ""
    client = DeepgramClient(api_key=api_key)

    with client.listen.v2.connect(
        model=model,
        encoding="linear16",
        sample_rate=SAMPLE_RATE,
    ) as socket:

        def _receive_loop() -> None:
            nonlocal final_transcript
            try:
                for message in socket:
                    if isinstance(message, ListenV2TurnInfo):
                        text = message.transcript.strip()
                        if text:
                            final_transcript = text
                            print(_CLEAR + text, end="", flush=True)
                            if on_partial:
                                on_partial(text)
            except Exception:
                pass

        recv_thread = threading.Thread(target=_receive_loop, daemon=True)
        recv_thread.start()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=CHUNK_FRAMES,
            callback=_audio_callback,
        ):
            while not stop_event.is_set():
                try:
                    chunk = audio_queue.get(timeout=0.1)
                    socket.send_media(chunk)
                except queue.Empty:
                    pass

            while not audio_queue.empty():
                try:
                    chunk = audio_queue.get_nowait()
                    socket.send_media(chunk)
                except queue.Empty:
                    break

        socket.send_close_stream()
        recv_thread.join(timeout=5.0)
        key_thread.join(timeout=1.0)

    print(flush=True)
    return final_transcript
