#!/usr/bin/env python3
"""Voice-first note capture with Deepgram streaming transcription.

Usage:
    python -m voice_transcribe.voice          # run as module
    voice                              # if installed via pyproject.toml scripts

Controls:
    Press SPACE           → start a voice turn
    Press SPACE again     → stop that turn
    Ctrl+C                → finalize the live session markdown and quit
"""

from voice_transcribe.cli import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
