# Deepgram Implementation Plan

This document describes the planned rewrite from the current mixed local-STT architecture to a Deepgram-only, conversation-first orchestrator model.

This is intentionally a rewrite plan, not a code diff. The goal is to simplify the system around:
- one user-facing conversation loop
- Deepgram-only STT
- text-only LLM replies
- structured protocol-driven orchestration
- agent execution through the same conversation model

## Product decisions locked for this rewrite

- STT becomes Deepgram-only
- Assistant replies remain text-only
- `SPACE` starts a conversation session
- any key ends the active conversation session
- one user-facing conversation loop
- no local Whisper/MLX/Granite fallback
- no TTS
- protocol-driven orchestrator is the primary control plane

## Rewrite goals

1. Eliminate the current batch/rolling-window STT model
2. Replace backend-selected STT with a single Deepgram streaming path
3. Replace the current record/review/add/correct loop with a conversation orchestrator
4. Keep markdown save and diagram generation, but invoke them through the new orchestrator
5. Simplify CLI, config, dependencies, and tests around the new model

## High-level migration phases

### Phase A: Contracts and architecture

1. Finalize `protocol.md`
- validate protocol fields
- validate state transitions
- validate agent result contract
- define parse/repair behavior

2. Define orchestrator runtime state
- `conversation_state`
- `current_plan`
- `confirmation_status`
- `active_agent`
- `agent_context`
- `suspended_agent_state`
- conversation history

3. Define Deepgram event model
- partial transcript event
- finalized transcript event
- connection error event

Note:
- v1 uses explicit push-to-talk session boundaries (`SPACE` starts, any key ends)
- Deepgram endpoint/turn-finished detection is not part of the initial design
- the streaming client should focus on live partial transcript updates and a final transcript for the active session

4. Decide completion semantics
- `completed` triggers `process_and_save(...)`
- save remains orchestrator-owned, not an LLM agent

### Phase B: New runtime path

1. Implement Deepgram streaming client
2. Implement orchestrator loop
3. Implement protocol parse/repair
4. Implement terminal rendering for partial/final user turns and assistant text

### Phase C: Reconnect existing features

1. Reconnect markdown generation and save
2. Adapt the existing diagram agent to the new agent result contract
3. Reconnect diagram agent through the orchestrator
4. Reconnect vector indexing after save
5. Keep per-session directory output

### Phase D: Cleanup and consolidation

1. Remove old STT backend machinery
2. Rewrite CLI/config around Deepgram-only behavior
3. Rewrite tests around protocol/orchestrator/Deepgram events
4. Update README/PLAN to reflect final architecture
5. Delete dead code

## File-by-file plan

## Keep with adaptation

### `src/voice_transcribe/llm.py`

Keep:
- backend execution helpers
- `claude` / `ollama` integration

Adapt:
- add helper for protocol-only JSON prompting if needed
- possibly add structured repair prompt helper

### `src/voice_transcribe/storage.py`

Keep:
- markdown generation and save
- session directory behavior
- vector indexing hook

Adapt:
- make save explicitly orchestrator-triggered on `completed`
- ensure return values fit orchestrator completion flow

### `src/voice_transcribe/prompts.py`

Keep:
- prompt loading/rendering for markdown generation

Adapt only if protocol-driven completion needs additional prompt variants.

### `src/voice_transcribe/vector.py`

Keep:
- summary-only embedding/indexing
- similarity color assignment

No major redesign needed.

### `src/voice_transcribe/agents/diagram.py`

Keep conceptually:
- diagram agent remains valuable

Adapt:
- return structured agent result contract (`success | error | needs_input`)
- support orchestrator-driven suspend/resume cleanly

### `src/voice_transcribe/tools/d2.py`

Keep conceptually:
- D2 render wrapper

Adapt only if richer structured error payloads are useful.

### `src/voice_transcribe/loops/agent.py`

Keep initially:
- generic agent loop is already useful for intra-agent tool use

Adapt:
- ensure agent result contract is compatible with orchestrator expectations
- support `needs_input` cleanly

Important:
- `loops/agent.py` is for agent-internal tool use, such as the diagram agent iterating on D2 renders
- it is not the orchestrator loop
- the orchestrator uses the structured protocol in `protocol.md`, which is a different LLM interaction pattern

## Rewrite heavily

### `src/voice_transcribe/audio.py`

Current role:
- rolling-window local transcription
- keyboard-controlled recording session

Rewrite into:
- microphone stream manager
- Deepgram audio transport integration
- terminal partial transcript rendering
- session start/stop control

Things to remove:
- rolling decode cadence
- overlap/dedupe logic
- batch-oriented transcript assembly

### `src/voice_transcribe/cli.py`

Current role:
- old STT flag parsing
- current top-level flow
- post-save diagram flow

Rewrite into:
- minimal Deepgram + LLM config parsing
- conversation session bootstrap
- orchestrator startup
- session completion/exit handling

Things to remove:
- old STT backend choices
- old post-save diagram flow as special-case CLI behavior

### `src/voice_transcribe/loops/conversational.py`

Current role:
- record
- review
- add/correct/exit

Likely outcome:
- remove or replace

Reason:
- this loop is tied to the old clip-based UX
- new model is a continuous conversation session with internal state

## Add new modules

### `src/voice_transcribe/deepgram_stream.py`

Responsibilities:
- open websocket connection to Deepgram
- stream PCM audio from mic
- receive partial transcript events
- receive finalized transcript for the active push-to-talk session
- surface structured events to the orchestrator
- handle connection errors/retries

### `src/voice_transcribe/orchestrator.py`

Responsibilities:
- hold orchestrator runtime state
- accept finalized user turns
- call LLM
- parse and validate protocol output
- run repair loop on invalid protocol output
- decide state transitions
- trigger agents
- resume suspended agents
- trigger completion/save

### `src/voice_transcribe/protocol_models.py`

Responsibilities:
- define code-level models for:
  - LLM protocol response
  - orchestrator state enums
  - agent result contract
- centralize validation rules

Optional names:
- `models.py`
- `schemas.py`

But there should be one clear place for protocol models.

## Simplify configuration

### `src/voice_transcribe/config.py`

Remove or deprecate:
- Whisper defaults
- MLX defaults
- Granite defaults

Add:
- Deepgram model default
- protocol repair retry count
- conversation session timeout/config
- terminal rendering constants
- API/environment variable names

## CLI and env cleanup

## Remove old CLI flags

Likely remove:
- `--stt-backend`
- `--whisper-model`
- `--mlx-model`
- `--granite-model`

## Add new CLI flags

Possible replacements:
- `--deepgram-model`
- `--debug-protocol`

Likely env vars:
- `DEEPGRAM_API_KEY`
- `VOICE_DEEPGRAM_MODEL`
- `VOICE_LLM_BACKEND`
- `VOICE_OLLAMA_MODEL`

## Testing plan

## Tests to remove or rewrite

### `tests/test_cli.py`

Rewrite:
- remove old STT-backend expectations
- add Deepgram-specific config expectations
- add clear validation expectations for missing `DEEPGRAM_API_KEY`

### Old loop-oriented tests

Any tests that assume:
- clip recording model
- add/correct/exit flow
- Whisper/MLX/Granite selection

should be removed or rewritten.

## Tests to keep

Keep with minimal changes:
- `tests/test_prompts.py`
- `tests/test_storage.py`
- `tests/test_vector.py`
- `tests/test_index_cmd.py`

These remain useful unless the save contract changes dramatically.

## Tests to add

### Protocol tests
- valid protocol parse
- invalid enum rejection
- missing field rejection
- repair prompt retry behavior

### Orchestrator tests
- `discussing -> discussing`
- `discussing -> confirming`
- `confirming -> executing_agent`
- `executing_agent -> resolving_agent_question`
- `resolving_agent_question -> executing_agent`
- `resolving_agent_question -> discussing`
- `completed -> save`

### Deepgram client tests
- partial transcript event handling
- finalized turn event handling
- session open/close boundary handling (push-to-talk, not Deepgram VAD)
- disconnect / reconnect behavior

### Agent suspend/resume tests
- diagram agent returns `needs_input`
- orchestrator stores suspended state
- user answer resumes same agent

## Dependency changes

## Add

- Deepgram SDK or websocket client dependency

## Remove later

Once migration is complete, remove unused STT dependencies such as:
- `faster-whisper`
- `transformers` if only used for old STT path
- `sentencepiece` if only used for old STT path
- `torchaudio` if only used for Granite Speech

Important:
- `torch` stays unless the embedding stack changes
- `vector.py` uses `sentence-transformers`, which depends on `torch`
- do not plan on removing `torch` as part of this rewrite unless embeddings are replaced too

## Documentation work

## `README.md`

Rewrite after migration:
- remove Whisper/MLX/Granite setup
- add Deepgram setup
- explain conversation session model
- explain single conversation-loop orchestrator
- document completion/diagram behavior

## `PLAN.md`

Update once rewrite begins:
- current state changes materially
- Deepgram rewrite becomes active work
- old STT assumptions become historical

## `protocol.md`

Keep as source of truth for:
- LLM output protocol
- agent result contract
- transition rules
- repair rules

## Recommended migration strategy

Use a parallel path first.

Meaning:
- add new Deepgram + orchestrator modules
- keep the current flow temporarily
- cut over once the new path is working
- then remove old code decisively

This reduces the risk of getting stuck mid-rewrite with a half-functional main path.

## Suggested execution order

1. Commit current docs/spec checkpoint
2. ✅ Add `protocol_models.py`
3. ✅ Add `deepgram_stream.py`
4. ✅ Add `orchestrator.py`
5. ✅ Add a minimal new conversation-session runner
6. ✅ Route finalized Deepgram turns into the orchestrator
7. ✅ Implement completion → `process_and_save(...)`
8. ✅ Adapt diagram agent to the new `success | error | needs_input` result contract (via adapter in conversation_session.py)
9. ✅ Reconnect diagram agent through orchestrator
10. ✅ Rewrite tests
11. ✅ Remove old STT/backends and old loop code
12. Rewrite README and finalize cleanup

## Early startup validation

`cli.py` should validate required Deepgram configuration before attempting any connection.

At minimum:
- check that `DEEPGRAM_API_KEY` is present
- fail fast with a clear error message if it is missing

This should happen at startup, not inside the streaming client after audio capture has already begun.

## Definition of done

The rewrite is complete when:
- the app uses Deepgram as the only STT backend
- the user experiences one continuous conversation session
- the LLM returns structured protocol output every turn
- the orchestrator owns all state transitions
- markdown save happens on `completed`
- diagram generation runs through the same orchestrator
- suspend/resume works for agents
- old batch/rolling-window transcription code is gone
- docs and tests reflect the new architecture
