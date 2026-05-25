# PLAN

## Current
- Rule for updates: when a phase is finished, move it from `Planned` to `Completed` with completion notes/date.

- Runtime entrypoints:
  - Main command: `voice`
  - Module run: `python -m voice_transcribe.voice`

- Current code structure:
  - `voice.py` — thin entrypoint only; imports `main` from `cli.py`
  - `cli.py` — argument parsing, config display, Deepgram API key validation, handoff into the live conversation session
  - `deepgram_stream.py` — push-to-talk Deepgram streaming (`SPACE` starts a turn, `SPACE` ends it)
  - `audio.py` — terminal display helpers
  - `llm.py` — LLM backend runners (`claude`, `openai`, `ollama`)
  - `prompts.py` — `process_prompt.md` loading and template rendering
  - `storage.py` — live markdown session artifact creation, update, finalize, and vector indexing
  - `vector.py` — LanceDB vector index for saved markdown summaries
  - `index_cmd.py` — `voice-index` CLI for backfilling existing transcripts
  - `config.py` — constants and defaults
  - `process_prompt.md` — prompt template that turns user source notes into structured markdown
  - `loops/conversation_session.py` — markdown-centered live voice loop
  - `loops/agent.py` — generic fenced-JSON agent loop used inside concrete agents
  - `agents/diagram.py` — diagram-generation agent that iterates on D2 syntax until render succeeds
  - `tools/d2.py` — D2 CLI wrapper used by the diagram agent

- Current STT behavior:
  - STT is Deepgram-only
  - active model default: `flux-general-en`
  - user controls turn boundaries explicitly:
    - `SPACE` starts a turn
    - `SPACE` ends that turn
    - `Ctrl+C` finalizes and exits the whole session

- Current LLM behavior:
  - LLM backends supported: `claude`, `openai`, and `ollama`
  - assistant replies are text-only
  - markdown generation uses `process_prompt.md`
  - live assistant replies are plain text, not protocol JSON

- Current output/local directory behavior:
  - the app creates a live durable session directory under `~/transcript/session_<timestamp>/`
  - a live `session.md` is refreshed from user speech after each turn
  - only user speech is used to build the base markdown artifact
  - generated diagrams are embedded back into the same markdown under `## Generated Artifacts`
  - on finalize, the session directory is renamed using an LLM-generated 5-word filename summary

- Current markdown artifact model:
  - `## Source Notes` is the durable base
  - `## Processed Transcript` is cleaned from the same user speech
  - `## Actions / Follow-ups` is derived from the same notes
  - `## Generated Artifacts` contains derived outputs such as diagrams

- Current CLI/options state:
  - `--llm-backend {claude,openai,ollama}`
  - `--ollama-model ...`
  - environment variable overrides:
    - `DEEPGRAM_API_KEY`
    - `VOICE_LLM_BACKEND`
    - `VOICE_OLLAMA_MODEL`

- Current known issues/constraints:
  - the markdown-centered loop is simpler than the earlier protocol/orchestrator design, but tool triggering is still heuristic rather than deeply structured
  - diagrams are currently the only embedded derived artifact
  - scheduling/Drive/screen-context flows are not integrated yet
  - live audio + TTY interaction remains lightly unit-tested due to hardware/terminal constraints

## Planned

### Phase 4: Artifact-Driven Tool Expansion
- Keep `session.md` as the shared durable base for iteration.
- Add more tool flows that operate directly on that artifact rather than on a separate protocol state.
- Candidate next tools:
  - appointment / calendar creation
  - related-task lookup
  - screen-context capture as an optional tool
- For each tool:
  - read the current markdown artifact
  - produce a derived output
  - embed or append the result back into the markdown
  - show the updated artifact to the user
- Exit criteria: multiple tools can enrich the same markdown artifact without overwriting the user’s source notes.

### Phase 5: Cross-Session Search and Recall
- Search across prior markdown sessions and related artifacts.
- Support:
  - semantic similarity search over session summaries
  - lookup of sessions containing diagrams or appointments
  - reuse of older context when starting new work
- Keep this as an indexing/retrieval problem first; do not introduce LangGraph unless multi-step orchestration becomes hard.
- Exit criteria: easy retrieval of similar prior sessions and related action items.

### Phase 6: Google Drive Snapshot Sync
- Add Drive client for upload/download of vector snapshot + manifest
- Sync flow: download latest snapshot -> update locally -> upload new snapshot
- Add lock/version checks to avoid overwrite races
- Add local-first fallback when Drive is unavailable
- Exit criteria: reliable snapshot round-trip and conflict handling

### Phase 7: Calendar Actions
- Parse `## Actions / Follow-ups` from the current markdown artifact
- If one or more actions exist, offer calendar creation
- Event title: action item text
- Event description:
  - link back to the markdown artifact
  - relevant context snippet from the session
- Exit criteria: confirmed calendar creation from artifact-derived action items

### Phase 8: Hardening and Operations
- Add observability for session save, diagram generation, indexing, and future tool runs
- Add retry/backoff/resume for API failures
- Expand integration tests with mocked external dependencies
- Document runbook (setup, backup/restore, rollback)
- Exit criteria: operationally stable artifact-centered workflow

## Optional / Exploratory

### UI Layer (not scheduled)
The tool is currently terminal-only. A UI would replace raw keypress control with buttons and panels while keeping the same core artifact model:

- start/stop voice turns
- show the current `session.md`
- preview generated diagrams
- browse similar prior sessions

### LangGraph / LangChain (not scheduled)
Current direction is to stay framework-light until workflow complexity truly demands more orchestration.

If adopted later, it should be for:
- durable multi-step tool workflows
- explicit resume/checkpoint behavior
- more complex cross-tool coordination

Not for:
- basic session state
- basic markdown persistence
- simple cross-session search

## Handoff Notes (LLM Resume)
- Source of truth: this file
- The markdown file is the durable session artifact
- Keep user source notes separate from derived artifacts
- Do not reintroduce a heavy protocol/orchestrator layer unless real complexity forces it
- Add structure in code only where it materially reduces ambiguity

## Completed

### Structural Simplification: Markdown-Centered Session Artifact — 2026-05-25
- Removed the standalone `protocol.md` design and the heavier protocol/orchestrator runtime path.
- Removed `protocol_models.py`, `orchestrator.py`, and the protocol test suite.
- Replaced the active session flow with a markdown-centered loop in `loops/conversation_session.py`.
- The live loop now:
  - captures one Deepgram voice turn at a time
  - appends only user speech into the durable session artifact
  - refreshes `session.md` after each turn
  - finalizes the session on `Ctrl+C` or explicit finish phrases
- Diagram generation now uses the current markdown artifact as the source of truth and embeds rendered diagrams back into that same file.
- `storage.py` now supports:
  - live session directory creation
  - session markdown refresh
  - final rename/index on completion

### Phase 3: Diagram Agent — 2026-05-23
- `storage.process_and_save(...)` was originally updated to write a self-contained session directory under `~/transcript/<name>/` instead of a flat markdown file
- `loops/agent.py` implemented: generic fenced-JSON tool-call loop for LLM-driven tool use
- `tools/d2.py` implemented: renders D2 syntax from stdin to an output image path and returns structured success/error results
- `agents/diagram.py` implemented: diagram-generation agent that iterates on D2 syntax and retries renders up to a bounded attempt count
- The diagram workflow survives the simplification and now works against the live markdown artifact instead of a protocol-controlled execution path

### Phase 2: Summary-Only Vector Foundation (Local) — 2026-03-04
- `vector.py` fully implemented: LanceDB storage, sentence-transformers embeddings (`all-MiniLM-L6-v2`, 384-dim)
- Embeds only `## Summary` section (robust regex extraction)
- `doc_id` = `sha256(summary_text)` — content-addressed, stable across file renames
- Schema: `doc_id`, `path`, `created_at`, `updated_at`, `embedding_model`, `dictionary_version` (0), `color_hex`, `color_index`, `vector`
- Color assignment: nearest-neighbor cosine search (excluding self by `doc_id`); threshold 0.82; else least-used palette slot from 64-color HSL palette
- `search(query_text, top_k, threshold)` public API added
- `voice-index` console script added (`index_cmd.py`) for backfilling existing transcripts
- DB location: `~/.voice_transcribe/index.lancedb`

### Phase 1: Refactor Without Behavior Change — 2026-03-04
- Split `voice.py` monolith into: `config.py`, `audio.py`, `llm.py`, `prompts.py`, `storage.py`, `cli.py`
- `voice.py` is now a thin entrypoint: imports and re-exports `main` from `cli.py`
- `pyproject.toml` entry point unchanged (`voice_transcribe.voice:main` still resolves correctly)
