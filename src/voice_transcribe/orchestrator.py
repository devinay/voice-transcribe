"""Orchestrator: owns all state transitions, LLM protocol calls, agent dispatch,
and suspend/resume for the single conversation loop.

Usage:
    orch = Orchestrator(llm_backend="claude", ollama_model="qwen2.5:7b-instruct")
    result = orch.turn(user_text)  # call once per finalized user utterance
    print(result.assistant_message)
    if result.orchestrator.next_action == NextAction.TRIGGER_AGENT:
        ...
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from voice_transcribe.llm import run_llm_prompt
from voice_transcribe.protocol_models import (
    AgentName,
    AgentResult,
    AgentStatus,
    LLMProtocolResponse,
    NextAction,
    OrchestratorState,
    ProtocolError,
    parse_agent_result,
    parse_llm_response,
)

_PROTOCOL_REPAIR_PROMPT = """\
Your last response did not match the required JSON schema.
Return exactly one valid JSON object with these fields:
assistant_message, orchestrator.state, orchestrator.next_action,
orchestrator.needs_confirmation, orchestrator.plan_summary,
orchestrator.agent_to_trigger, orchestrator.agent_input,
orchestrator.question_for_user.
Do not include markdown fences or explanation.

Invalid response was:
{invalid_response}
"""

_SYSTEM_PROMPT = """\
You are a voice conversation orchestrator. Your job is to help the user plan and execute tasks through voice conversation.

You MUST return a single valid JSON object on every turn. No prose, no markdown fences, only the JSON object.

JSON schema:
{
  "assistant_message": "<text shown to user>",
  "orchestrator": {
    "state": "discussing | confirming | executing_agent | resolving_agent_question | completed",
    "next_action": "continue_discussion | wait_for_user | trigger_agent | complete",
    "needs_confirmation": true | false,
    "plan_summary": "<short summary of current plan, or null>",
    "agent_to_trigger": "diagram | null",
    "agent_input": { ... } | null,
    "question_for_user": "<single question for the user, or null>"
  }
}

Rules:
- assistant_message is shown to the user. Keep it concise and conversational.
- state and next_action must be valid enum values.
- When next_action = trigger_agent: agent_to_trigger and agent_input must be non-null.
- When state = resolving_agent_question: question_for_user and agent_to_trigger must be non-null.
- When state = completed: next_action must be complete.
- agent_to_trigger allowed values: diagram (v1 only).
- For diagram agent_input, include: {"description": "<what to draw>"}.
- Do not embed machine-only instructions in assistant_message.
"""

_MAX_REPAIR_ATTEMPTS = 2


@dataclass
class OrchestratorRuntimeState:
    conversation_state: OrchestratorState = OrchestratorState.DISCUSSING
    current_plan: str | None = None
    confirmation_status: str = "none"
    active_agent: AgentName | None = None
    agent_context: dict[str, Any] = field(default_factory=dict)
    suspended_agent_state: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, str]] = field(default_factory=list)


class Orchestrator:
    def __init__(self, llm_backend: str, ollama_model: str) -> None:
        self._llm_backend = llm_backend
        self._ollama_model = ollama_model
        self._state = OrchestratorRuntimeState()
        self._agent_registry: dict[AgentName, Any] = {}

    def register_agent(self, name: AgentName, agent_fn: Any) -> None:
        """Register a callable agent. agent_fn(input, resume_context, user_answer) -> dict."""
        self._agent_registry[name] = agent_fn

    # ------------------------------------------------------------------
    # Public turn entry point
    # ------------------------------------------------------------------

    def turn(self, user_text: str) -> LLMProtocolResponse:
        """Process one finalized user utterance. Returns the parsed LLM response.

        Handles state updates and agent dispatch automatically when the response
        indicates trigger_agent or complete.
        """
        self._state.history.append({"role": "user", "content": user_text})
        response = self._call_llm_with_repair()
        self._apply_state_update(response)
        self._state.history.append({"role": "assistant", "content": response.assistant_message})
        return response

    def dispatch_agent(self, name: AgentName, agent_input: dict[str, Any]) -> AgentResult:
        """Call the registered agent and handle its result contract."""
        agent_fn = self._agent_registry.get(name)
        if agent_fn is None:
            return AgentResult(status=AgentStatus.ERROR, error=f"no agent registered for {name.value!r}")
        try:
            raw = agent_fn(
                agent_input=agent_input,
                resume_context=self._state.suspended_agent_state.get(name.value),
                user_answer=None,
            )
            return parse_agent_result(raw)
        except ProtocolError as exc:
            return AgentResult(status=AgentStatus.ERROR, error=f"agent result contract violation: {exc}")
        except Exception as exc:
            return AgentResult(status=AgentStatus.ERROR, error=str(exc))

    def resume_agent(self, name: AgentName, user_answer: str) -> AgentResult:
        """Resume a suspended agent with the user's answer."""
        agent_fn = self._agent_registry.get(name)
        if agent_fn is None:
            return AgentResult(status=AgentStatus.ERROR, error=f"no agent registered for {name.value!r}")
        resume_ctx = self._state.suspended_agent_state.pop(name.value, None)
        try:
            raw = agent_fn(
                agent_input=None,
                resume_context=resume_ctx,
                user_answer=user_answer,
            )
            return parse_agent_result(raw)
        except ProtocolError as exc:
            return AgentResult(status=AgentStatus.ERROR, error=f"agent result contract violation: {exc}")
        except Exception as exc:
            return AgentResult(status=AgentStatus.ERROR, error=str(exc))

    def suspend_agent(self, name: AgentName, agent_result: AgentResult) -> None:
        """Store agent suspension state after a needs_input result."""
        if agent_result.resume_context:
            self._state.suspended_agent_state[name.value] = agent_result.resume_context
        self._state.active_agent = name
        self._state.conversation_state = OrchestratorState.RESOLVING_AGENT_QUESTION

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self) -> str:
        context_block = {
            "conversation_state": self._state.conversation_state.value,
            "current_plan": self._state.current_plan,
            "confirmation_status": self._state.confirmation_status,
            "active_agent": self._state.active_agent.value if self._state.active_agent else None,
            "question_for_user": (
                self._state.suspended_agent_state.get(
                    self._state.active_agent.value, {}
                ).get("pending_question")
                if self._state.active_agent
                else None
            ),
        }
        history_lines = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in self._state.history
        )
        return (
            f"{_SYSTEM_PROMPT}\n\n"
            f"Orchestrator context:\n{json.dumps(context_block, indent=2)}\n\n"
            f"Conversation so far:\n{history_lines}\n\n"
            "Assistant (return JSON only):"
        )

    def _call_llm_with_repair(self) -> LLMProtocolResponse:
        prompt = self._build_prompt()
        print(f"\n{'─' * 60}\n[PROMPT]\n{prompt}\n{'─' * 60}\n", flush=True)
        raw = run_llm_prompt(prompt, self._llm_backend, self._ollama_model)

        for attempt in range(_MAX_REPAIR_ATTEMPTS + 1):
            try:
                return parse_llm_response(raw)
            except ProtocolError as exc:
                if attempt == _MAX_REPAIR_ATTEMPTS:
                    raise
                repair_prompt = _PROTOCOL_REPAIR_PROMPT.format(invalid_response=raw)
                print(f"\n  [orchestrator] protocol repair attempt {attempt + 1}: {exc}", flush=True)
                raw = run_llm_prompt(repair_prompt, self._llm_backend, self._ollama_model)

        raise ProtocolError("repair exhausted")

    def _apply_state_update(self, response: LLMProtocolResponse) -> None:
        d = response.orchestrator
        self._state.conversation_state = d.state

        # Update plan only in discussing/confirming states (not during agent execution)
        if d.plan_summary and d.state in (OrchestratorState.DISCUSSING, OrchestratorState.CONFIRMING):
            self._state.current_plan = d.plan_summary

        if d.state == OrchestratorState.CONFIRMING and d.needs_confirmation:
            self._state.confirmation_status = "pending"
        elif d.next_action == NextAction.TRIGGER_AGENT and d.agent_to_trigger:
            self._state.confirmation_status = "confirmed"
            self._state.active_agent = d.agent_to_trigger
        elif d.state == OrchestratorState.COMPLETED:
            self._state.confirmation_status = "completed"
