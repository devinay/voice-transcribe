#!/usr/bin/env python3
"""Voice recorder with local Whisper transcription.

Usage:
    python -m voice_transcribe.voice          # run as module
    voice                              # if installed via pyproject.toml scripts

Controls:
    Press SPACE           → start recording (words appear live as you speak)
    Press any key         → stop recording
    Ctrl+C               → quit
"""

from voice_transcribe.cli import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
