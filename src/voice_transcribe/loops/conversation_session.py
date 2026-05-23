"""Single conversation-loop session runner.

This is the new top-level entry point for the voice conversation. It:
- runs one push-to-talk session per user turn (SPACE=start, any key=stop)
- feeds the finalized transcript into the orchestrator
- prints the assistant reply
- dispatches agents when the orchestrator emits trigger_agent
- handles agent suspend/resume through the same conversation loop
- triggers process_and_save on completed

Replaces the old loops/conversational.py clip-based UX.
"""
from __future__ import annotations

from voice_transcribe.deepgram_stream import run_push_to_talk_session
from voice_transcribe.orchestrator import Orchestrator
from voice_transcribe.protocol_models import AgentName, AgentStatus, NextAction, OrchestratorState


_GREEN = "\033[32m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def run(
    api_key: str,
    deepgram_model: str,
    llm_backend: str,
    ollama_model: str,
) -> None:
    """Run the full conversation session until completion or Ctrl+C."""
    from voice_transcribe import agents  # noqa: F401 — side-effect: loads agent package
    from voice_transcribe.agents import diagram as diagram_agent
    from voice_transcribe.protocol_models import ProtocolError
    from voice_transcribe.storage import process_and_save

    orch = Orchestrator(llm_backend=llm_backend, ollama_model=ollama_model)

    # Register diagram agent with the orchestrator result contract adapter
    def _diagram_agent_fn(
        agent_input: dict | None,
        resume_context: dict | None,
        user_answer: str | None,
    ) -> dict:
        import pathlib
        from voice_transcribe.config import TRANSCRIPT_DIR

        description = (agent_input or {}).get("description", "") or (resume_context or {}).get("description", "")
        output_path = TRANSCRIPT_DIR / "tmp_diagram.png"

        result = diagram_agent.run(
            description=description,
            output_path=pathlib.Path(output_path),
            llm_backend=llm_backend,
            ollama_model=ollama_model,
        )
        if result.get("success"):
            return {
                "status": "success",
                "result": {
                    "image_path": str(result.get("image_path", output_path)),
                    "syntax": result.get("syntax", ""),
                },
                "error": None,
                "question_for_user": None,
                "resume_context": None,
            }
        return {
            "status": "error",
            "result": None,
            "error": result.get("error", "unknown error"),
            "question_for_user": None,
            "resume_context": None,
        }

    orch.register_agent(AgentName.DIAGRAM, _diagram_agent_fn)

    print("\n  Starting conversation. Ctrl+C to exit.\n")

    while True:
        try:
            # One push-to-talk turn
            user_text = run_push_to_talk_session(
                api_key=api_key,
                model=deepgram_model,
            )
        except KeyboardInterrupt:
            print("\n  (conversation ended)", flush=True)
            break

        if not user_text.strip():
            print("  (no speech detected)", flush=True)
            continue

        print(f"\n  You: {user_text}\n", flush=True)

        # Check if we're resuming a suspended agent
        if orch._state.conversation_state == OrchestratorState.RESOLVING_AGENT_QUESTION and orch._state.active_agent:
            agent_result = orch.resume_agent(orch._state.active_agent, user_text)
            _handle_agent_result(orch, agent_result, llm_backend, ollama_model)
            continue

        # Normal orchestrator turn
        try:
            response = orch.turn(user_text)
        except ProtocolError as exc:
            print(f"\n  [error] protocol repair failed: {exc}", flush=True)
            print("  Please restate your request.", flush=True)
            continue

        print(f"\n  {_BOLD}Assistant:{_RESET} {response.assistant_message}\n", flush=True)

        d = response.orchestrator

        if d.next_action == NextAction.COMPLETE or d.state == OrchestratorState.COMPLETED:
            _do_completion(orch, llm_backend, ollama_model)
            break

        if d.next_action == NextAction.TRIGGER_AGENT and d.agent_to_trigger:
            agent_result = orch.dispatch_agent(d.agent_to_trigger, d.agent_input or {})
            _handle_agent_result(orch, agent_result, llm_backend, ollama_model)

        # For discussing/confirming/resolving states, loop back for next user turn


def _handle_agent_result(orch: Orchestrator, agent_result, llm_backend: str, ollama_model: str) -> None:
    from voice_transcribe.protocol_models import ProtocolError

    if agent_result.status == AgentStatus.SUCCESS:
        orch._state.active_agent = None
        # Feed result back into orchestrator as a system message
        result_summary = f"Agent completed successfully. Result: {agent_result.result}"
        try:
            response = orch.turn(result_summary)
            print(f"\n  {_BOLD}Assistant:{_RESET} {response.assistant_message}\n", flush=True)
        except ProtocolError:
            pass

    elif agent_result.status == AgentStatus.ERROR:
        print(f"\n  [agent error] {agent_result.error}", flush=True)
        error_msg = f"Agent failed with error: {agent_result.error}"
        try:
            response = orch.turn(error_msg)
            print(f"\n  {_BOLD}Assistant:{_RESET} {response.assistant_message}\n", flush=True)
        except ProtocolError:
            pass

    elif agent_result.status == AgentStatus.NEEDS_INPUT:
        orch.suspend_agent(orch._state.active_agent, agent_result)
        print(f"\n  {_BOLD}Assistant:{_RESET} {agent_result.question_for_user}\n", flush=True)


def _do_completion(orch: Orchestrator, llm_backend: str, ollama_model: str) -> None:
    from voice_transcribe.storage import process_and_save

    history = orch._state.history
    full_transcript = " ".join(
        m["content"] for m in history if m["role"] == "user"
    )
    if full_transcript.strip():
        print("\n  Processing and saving...", flush=True)
        session_dir = process_and_save(full_transcript, llm_backend, ollama_model)
        print(f"\n  Session saved → {session_dir}", flush=True)
    else:
        print("\n  (nothing to save)", flush=True)
