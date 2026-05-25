# Deepgram Implementation Notes

This document now describes the simplified Deepgram-first architecture that is actually in the repo.

The important change is conceptual:

- the system no longer revolves around a protocol-driven orchestrator
- it revolves around a durable markdown session artifact

## Core model

### User-facing loop

- `SPACE` starts one voice turn
- `SPACE` ends that turn
- Deepgram produces the transcript for that turn
- the user’s spoken content is appended to the live session artifact
- the assistant replies in text
- if the user requests a diagram, the diagram agent uses the markdown artifact as the source of truth

### Durable artifact

The session directory is created immediately:

```text
~/transcript/session_<timestamp>/
```

The live markdown file inside it is:

```text
session.md
```

That file is refreshed after each user turn from the accumulated user speech.

On finalize, the directory is renamed to a stable summary-based name and indexed.

## Why this replaced the protocol design

The earlier protocol/orchestrator design was structurally clean, but it was too heavy for the current product shape.

The markdown-centered model better matches the real goal:

- preserve what the user said
- let the agent operate on that artifact
- embed derived outputs back into it
- show the updated result immediately

## Current data ownership

### User speech owns

- `## Source Notes`
- the base transcript content
- the meaning of the session

### LLM processing owns

- `## Summary`
- `## Processed Transcript`
- `## Actions / Follow-ups`

### Tools/agents own

- `## Generated Artifacts`
- diagram images
- future appointment or screen-context outputs

## Current file responsibilities

### `src/voice_transcribe/deepgram_stream.py`

- push-to-talk Deepgram streaming
- live partial display
- final transcript per turn

### `src/voice_transcribe/loops/conversation_session.py`

- owns the live conversation session
- accumulates user turns
- refreshes `session.md`
- triggers diagram generation when the user asks for it
- finalizes the session on `Ctrl+C` or explicit finish phrases

### `src/voice_transcribe/storage.py`

- creates the live session directory
- rewrites `session.md` from user source notes
- appends generated artifact embeds
- renames/finalizes the session directory
- indexes the final markdown file

### `src/voice_transcribe/agents/diagram.py`

- uses the current markdown artifact as the source of truth
- generates D2 syntax
- renders the diagram
- returns the image path for embedding

## Current limitations

- diagram triggering is still heuristic and phrase-based
- only diagrams are integrated as embedded artifacts today
- there is not yet a generalized tool framework around the markdown artifact
- session finalization is explicit rather than fully conversational

## Near-term next steps

1. Validate the live Deepgram path thoroughly
2. Improve diagram edit ergonomics
3. Add artifact-aware follow-up tools
4. Add cross-session search and recall
5. Keep the system simple until real orchestration complexity appears

## Things deliberately not in scope right now

- LangGraph / LangChain
- a heavy JSON protocol between the LLM and runtime
- a separate orchestrator state machine
- continuous hands-free VAD-driven turn detection
- TTS

If those return later, they should be introduced only after the artifact-centered workflow stops being sufficient.
