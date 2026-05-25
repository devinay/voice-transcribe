# voice-transcribe

Voice-first notes with Deepgram streaming transcription, LLM-assisted markdown cleanup, local vector indexing, and optional D2 diagrams.

## Current model

The app now revolves around one durable session artifact:

- `SPACE` starts one voice turn
- `SPACE` again ends that turn
- only the user's spoken content is written into the live `session.md`
- the assistant replies in text
- if you ask for a diagram, the diagram agent uses `session.md` as the source of truth
- generated diagrams are embedded back into that same markdown file

This keeps the session grounded in the user’s notes instead of in agent chatter or protocol state.

## Install

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
uv pip install -e .
```

Required runtime dependencies:

- Deepgram API key for STT
- one LLM backend:
  - Claude via `ANTHROPIC_API_KEY`
  - OpenAI via `OPENAI_API_KEY`
  - or local Ollama

Optional:

- `d2` for diagram rendering

Install `d2` on macOS:

```bash
brew install d2
```

## Environment

Set Deepgram:

```bash
export DEEPGRAM_API_KEY=your_key_here
```

Choose one LLM path.

### Claude

```bash
export ANTHROPIC_API_KEY=your_key_here
voice --llm-backend claude
```

### OpenAI

```bash
export OPENAI_API_KEY=your_key_here
voice --llm-backend openai
```

### Ollama

On a local 8 GB M1 MacBook Air, `qwen2.5:7b-instruct` is the most practical default right now.

```bash
ollama serve
voice --llm-backend ollama --ollama-model qwen2.5:7b-instruct
```

## Usage

Show current config:

```bash
voice
```

Start a session:

```bash
voice --llm-backend claude
```

Controls:

- `SPACE` starts a turn
- `SPACE` again ends that turn
- `Ctrl+C` finalizes the session markdown and exits

You can also explicitly say things like:

- “save and exit”
- “finish this session”
- “draw a diagram for this”
- “update the diagram to show the queue”

## Session artifact

During a live session, the app maintains a live directory under:

```text
~/transcript/session_YYYYMMDD_HHMMSS/
```

Inside it:

```text
session.md
diagram_1.png
diagram_2.png
...
```

When the session is finalized, that directory is renamed using an LLM-generated 5-word summary, for example:

```text
~/transcript/api_queue_rollout_notes/
└── api_queue_rollout_notes.md
```

## Markdown structure

The markdown artifact is built from user speech and currently uses these sections:

```md
# Title

## Summary

## Source Notes

## Processed Transcript

## Actions / Follow-ups

## Generated Artifacts
```

Important rule:

- `Source Notes` come from the user’s spoken content only
- generated diagrams and future tool results go under `Generated Artifacts`

## Diagrams

If your turn asks for a diagram, the app:

1. reads the current `session.md`
2. uses it as the diagram source of truth
3. runs the D2 diagram agent
4. saves the rendered image into the session directory
5. embeds the image back into `session.md`
6. displays the updated markdown

This lets you iterate by speaking changes against the current artifact.

## Vector index

After final save, the `## Summary` section is embedded and stored in a local LanceDB index.

Current embedding model:

- `sentence-transformers/all-MiniLM-L6-v2`

Index location:

```text
~/.voice_transcribe/index.lancedb
```

Backfill old transcripts:

```bash
voice-index
```

## Code structure

```text
src/voice_transcribe/
├── voice.py                # thin entrypoint
├── cli.py                  # args, startup validation, bootstrap
├── deepgram_stream.py      # push-to-talk Deepgram streaming session
├── audio.py                # terminal display helpers
├── llm.py                  # Claude / OpenAI / Ollama runners
├── prompts.py              # processing prompt loader
├── process_prompt.md       # markdown build prompt
├── storage.py              # live session artifact + finalize/index helpers
├── vector.py               # LanceDB index
├── index_cmd.py            # voice-index
├── loops/
│   ├── conversation_session.py   # markdown-centered live loop
│   └── agent.py                  # generic intra-agent tool loop
├── agents/
│   └── diagram.py         # D2 diagram agent
└── tools/
    └── d2.py              # D2 renderer wrapper
```

## Tests

Run:

```bash
.venv/bin/pytest -q
```

Current test coverage includes:

- CLI defaults and env handling
- prompt loading
- session artifact save/finalize behavior
- vector indexing
- backfill indexing command

## Roadmap notes

Completed:

- Phase 1 refactor
- Phase 2 local vector indexing
- Phase 3 diagram generation
- Deepgram-only simplification
- markdown-centered live session artifact

Near-term direction:

- keep the markdown file as the durable base of iteration
- expand tool flows around that artifact
- let future tools append structured results back into the same markdown

Later ideas:

- search across prior markdown sessions and related appointments
- Drive sync for the vector snapshot
- Calendar creation from action items
- optional screen-context capture as a tool
