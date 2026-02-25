# voice-transcribe

A local, offline voice recorder with live Whisper transcription and automatic summarisation.

- Press **SPACE** to start recording — words appear on screen as you speak
- Press **SPACE** again to stop
- A summary is generated automatically using a local BART model
- Review the result, then choose to proceed, edit, append another clip, or exit

Everything runs locally — no audio leaves your machine.

## Requirements

- Python 3.11+
- A working microphone

## Setup

Install with [uv](https://github.com/astral-sh/uv) (recommended):

```bash
uv pip install -e ".[dev]"
```

Or with pip:

```bash
pip install -e ".[dev]"
```

On first run the Whisper (`medium.en`) and BART summarisation model weights are downloaded automatically.

## Usage

```bash
voice
```

Or run as a module:

```bash
python -m test_repo.voice
```

### Controls

| Key | Action |
|-----|--------|
| `SPACE` | Start recording |
| `SPACE` | Stop recording |
| `P` / Enter | Proceed with transcript |
| `E` | Edit the transcript manually |
| `A` | Append another recording clip |
| `X` | Exit |

## Models

| Component | Model | Notes |
|-----------|-------|-------|
| Transcription | `faster-whisper medium.en` | English-only, runs on CPU (int8) or GPU (float16) |
| Summarisation | `philschmid/bart-large-cnn-samsum` | Summarises up to ~900 words |

## Tests

```bash
pytest
```
