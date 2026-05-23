from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProtocolError(Exception):
    pass


class OrchestratorState(str, Enum):
    DISCUSSING = "discussing"
    CONFIRMING = "confirming"
    EXECUTING_AGENT = "executing_agent"
    RESOLVING_AGENT_QUESTION = "resolving_agent_question"
    COMPLETED = "completed"


class NextAction(str, Enum):
    CONTINUE_DISCUSSION = "continue_discussion"
    WAIT_FOR_USER = "wait_for_user"
    TRIGGER_AGENT = "trigger_agent"
    COMPLETE = "complete"


class AgentName(str, Enum):
    DIAGRAM = "diagram"


class AgentStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    NEEDS_INPUT = "needs_input"


@dataclass
class OrchestratorDirective:
    state: OrchestratorState
    next_action: NextAction
    needs_confirmation: bool
    plan_summary: str | None
    agent_to_trigger: AgentName | None
    agent_input: dict[str, Any] | None
    question_for_user: str | None


@dataclass
class LLMProtocolResponse:
    assistant_message: str
    orchestrator: OrchestratorDirective


@dataclass
class AgentResult:
    status: AgentStatus
    result: dict[str, Any] | None = field(default=None)
    error: str | None = field(default=None)
    question_for_user: str | None = field(default=None)
    resume_context: dict[str, Any] | None = field(default=None)


def parse_llm_response(raw: str) -> LLMProtocolResponse:
    """Parse and validate a raw LLM string into a LLMProtocolResponse.

    Strips markdown fences if present. Raises ProtocolError on any violation.
    """
    stripped = raw.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        stripped = "\n".join(lines[1:end])

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ProtocolError("response must be a JSON object")

    msg = data.get("assistant_message")
    if not msg or not isinstance(msg, str):
        raise ProtocolError("assistant_message must be a non-empty string")

    orch = data.get("orchestrator")
    if not isinstance(orch, dict):
        raise ProtocolError("orchestrator must be an object")

    try:
        state = OrchestratorState(orch.get("state"))
    except ValueError:
        raise ProtocolError(f"invalid state: {orch.get('state')!r}")

    try:
        next_action = NextAction(orch.get("next_action"))
    except ValueError:
        raise ProtocolError(f"invalid next_action: {orch.get('next_action')!r}")

    nc = orch.get("needs_confirmation")
    if not isinstance(nc, bool):
        raise ProtocolError("needs_confirmation must be a boolean")

    atr_raw = orch.get("agent_to_trigger")
    if atr_raw is None:
        agent_to_trigger = None
    else:
        try:
            agent_to_trigger = AgentName(atr_raw)
        except ValueError:
            raise ProtocolError(f"invalid agent_to_trigger: {atr_raw!r}")

    plan_summary = orch.get("plan_summary")
    agent_input = orch.get("agent_input")
    question_for_user = orch.get("question_for_user")

    directive = OrchestratorDirective(
        state=state,
        next_action=next_action,
        needs_confirmation=nc,
        plan_summary=plan_summary if isinstance(plan_summary, str) else None,
        agent_to_trigger=agent_to_trigger,
        agent_input=agent_input if isinstance(agent_input, dict) else None,
        question_for_user=question_for_user if isinstance(question_for_user, str) else None,
    )

    # Conditional invariants
    if next_action == NextAction.TRIGGER_AGENT:
        if directive.agent_to_trigger is None:
            raise ProtocolError("agent_to_trigger must be non-null when next_action=trigger_agent")
        if directive.agent_input is None:
            raise ProtocolError("agent_input must be non-null when next_action=trigger_agent")

    if state == OrchestratorState.RESOLVING_AGENT_QUESTION:
        if directive.question_for_user is None:
            raise ProtocolError("question_for_user must be non-null in resolving_agent_question")
        if directive.agent_to_trigger is None:
            raise ProtocolError("agent_to_trigger must be non-null in resolving_agent_question")

    if state == OrchestratorState.COMPLETED:
        if next_action != NextAction.COMPLETE:
            raise ProtocolError("next_action must be complete when state=completed")

    return LLMProtocolResponse(assistant_message=msg, orchestrator=directive)


def parse_agent_result(result: dict[str, Any]) -> AgentResult:
    """Validate and parse a raw agent result dict into an AgentResult.

    Raises ProtocolError if the result does not match the agent result contract.
    """
    if not isinstance(result, dict):
        raise ProtocolError("agent result must be a dict")

    try:
        status = AgentStatus(result.get("status"))
    except ValueError:
        raise ProtocolError(f"invalid agent status: {result.get('status')!r}")

    if status == AgentStatus.SUCCESS:
        r = result.get("result")
        if r is None:
            raise ProtocolError("result must be non-null for status=success")
        return AgentResult(status=status, result=r)

    if status == AgentStatus.ERROR:
        err = result.get("error")
        if not err or not isinstance(err, str):
            raise ProtocolError("error must be a non-empty string for status=error")
        return AgentResult(status=status, error=err)

    if status == AgentStatus.NEEDS_INPUT:
        q = result.get("question_for_user")
        if not q or not isinstance(q, str):
            raise ProtocolError("question_for_user must be non-empty for status=needs_input")
        rc = result.get("resume_context")
        if rc is None:
            raise ProtocolError("resume_context must be non-null for status=needs_input")
        return AgentResult(
            status=status,
            question_for_user=q,
            resume_context=rc,
            result=result.get("result"),
        )

    raise ProtocolError(f"unhandled status: {status}")
