import json
import pytest

from voice_transcribe.protocol_models import (
    AgentName,
    AgentStatus,
    NextAction,
    OrchestratorState,
    ProtocolError,
    parse_agent_result,
    parse_llm_response,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(**overrides) -> str:
    base = {
        "assistant_message": "Hello.",
        "orchestrator": {
            "state": "discussing",
            "next_action": "continue_discussion",
            "needs_confirmation": False,
            "plan_summary": None,
            "agent_to_trigger": None,
            "agent_input": None,
            "question_for_user": None,
        },
    }
    base.update(overrides)
    return json.dumps(base)


def _make_orch(**overrides) -> str:
    orch = {
        "state": "discussing",
        "next_action": "continue_discussion",
        "needs_confirmation": False,
        "plan_summary": None,
        "agent_to_trigger": None,
        "agent_input": None,
        "question_for_user": None,
    }
    orch.update(overrides)
    return json.dumps({"assistant_message": "Ok.", "orchestrator": orch})


# ---------------------------------------------------------------------------
# Valid protocol responses
# ---------------------------------------------------------------------------

def test_parse_valid_discussing():
    raw = _make_response()
    r = parse_llm_response(raw)
    assert r.assistant_message == "Hello."
    assert r.orchestrator.state == OrchestratorState.DISCUSSING
    assert r.orchestrator.next_action == NextAction.CONTINUE_DISCUSSION
    assert r.orchestrator.needs_confirmation is False


def test_parse_valid_confirming():
    raw = _make_orch(
        state="confirming",
        next_action="wait_for_user",
        needs_confirmation=True,
        question_for_user="Ready to proceed?",
    )
    r = parse_llm_response(raw)
    assert r.orchestrator.state == OrchestratorState.CONFIRMING
    assert r.orchestrator.needs_confirmation is True
    assert r.orchestrator.question_for_user == "Ready to proceed?"


def test_parse_valid_trigger_agent():
    raw = _make_orch(
        state="executing_agent",
        next_action="trigger_agent",
        agent_to_trigger="diagram",
        agent_input={"description": "A flow diagram"},
    )
    r = parse_llm_response(raw)
    assert r.orchestrator.next_action == NextAction.TRIGGER_AGENT
    assert r.orchestrator.agent_to_trigger == AgentName.DIAGRAM
    assert r.orchestrator.agent_input == {"description": "A flow diagram"}


def test_parse_valid_completed():
    raw = _make_orch(state="completed", next_action="complete")
    r = parse_llm_response(raw)
    assert r.orchestrator.state == OrchestratorState.COMPLETED
    assert r.orchestrator.next_action == NextAction.COMPLETE


def test_parse_strips_markdown_fences():
    inner = json.dumps({
        "assistant_message": "Hi.",
        "orchestrator": {
            "state": "discussing",
            "next_action": "continue_discussion",
            "needs_confirmation": False,
            "plan_summary": None,
            "agent_to_trigger": None,
            "agent_input": None,
            "question_for_user": None,
        },
    })
    fenced = f"```json\n{inner}\n```"
    r = parse_llm_response(fenced)
    assert r.assistant_message == "Hi."


# ---------------------------------------------------------------------------
# Invalid protocol responses
# ---------------------------------------------------------------------------

def test_invalid_json_raises():
    with pytest.raises(ProtocolError, match="invalid JSON"):
        parse_llm_response("not json")


def test_missing_assistant_message_raises():
    raw = json.dumps({"orchestrator": {"state": "discussing", "next_action": "continue_discussion", "needs_confirmation": False}})
    with pytest.raises(ProtocolError, match="assistant_message"):
        parse_llm_response(raw)


def test_empty_assistant_message_raises():
    raw = _make_response(assistant_message="")
    with pytest.raises(ProtocolError, match="assistant_message"):
        parse_llm_response(raw)


def test_invalid_state_raises():
    raw = _make_orch(state="flying")
    with pytest.raises(ProtocolError, match="invalid state"):
        parse_llm_response(raw)


def test_invalid_next_action_raises():
    raw = _make_orch(next_action="do_magic")
    with pytest.raises(ProtocolError, match="invalid next_action"):
        parse_llm_response(raw)


def test_needs_confirmation_not_bool_raises():
    raw = _make_orch(needs_confirmation="yes")
    with pytest.raises(ProtocolError, match="needs_confirmation"):
        parse_llm_response(raw)


def test_trigger_agent_missing_agent_name_raises():
    raw = _make_orch(
        state="executing_agent",
        next_action="trigger_agent",
        agent_to_trigger=None,
        agent_input={"description": "x"},
    )
    with pytest.raises(ProtocolError, match="agent_to_trigger"):
        parse_llm_response(raw)


def test_trigger_agent_missing_agent_input_raises():
    raw = _make_orch(
        state="executing_agent",
        next_action="trigger_agent",
        agent_to_trigger="diagram",
        agent_input=None,
    )
    with pytest.raises(ProtocolError, match="agent_input"):
        parse_llm_response(raw)


def test_resolving_missing_question_raises():
    raw = _make_orch(
        state="resolving_agent_question",
        next_action="wait_for_user",
        agent_to_trigger="diagram",
        question_for_user=None,
    )
    with pytest.raises(ProtocolError, match="question_for_user"):
        parse_llm_response(raw)


def test_completed_wrong_next_action_raises():
    raw = _make_orch(state="completed", next_action="continue_discussion")
    with pytest.raises(ProtocolError, match="next_action must be complete"):
        parse_llm_response(raw)


def test_invalid_agent_name_raises():
    raw = _make_orch(
        state="executing_agent",
        next_action="trigger_agent",
        agent_to_trigger="calendar",  # not in v1 enum
        agent_input={"description": "x"},
    )
    with pytest.raises(ProtocolError, match="invalid agent_to_trigger"):
        parse_llm_response(raw)


# ---------------------------------------------------------------------------
# Agent result contract
# ---------------------------------------------------------------------------

def test_agent_result_success():
    r = parse_agent_result({"status": "success", "result": {"image_path": "/tmp/x.png"}})
    assert r.status == AgentStatus.SUCCESS
    assert r.result == {"image_path": "/tmp/x.png"}


def test_agent_result_error():
    r = parse_agent_result({"status": "error", "error": "d2 not found"})
    assert r.status == AgentStatus.ERROR
    assert r.error == "d2 not found"


def test_agent_result_needs_input():
    r = parse_agent_result({
        "status": "needs_input",
        "question_for_user": "Deployment or request flow?",
        "resume_context": {"attempt_count": 1},
    })
    assert r.status == AgentStatus.NEEDS_INPUT
    assert r.question_for_user == "Deployment or request flow?"
    assert r.resume_context == {"attempt_count": 1}


def test_agent_result_invalid_status_raises():
    with pytest.raises(ProtocolError, match="invalid agent status"):
        parse_agent_result({"status": "unknown"})


def test_agent_result_success_null_result_raises():
    with pytest.raises(ProtocolError, match="result must be non-null"):
        parse_agent_result({"status": "success", "result": None})


def test_agent_result_error_empty_string_raises():
    with pytest.raises(ProtocolError, match="error must be a non-empty string"):
        parse_agent_result({"status": "error", "error": ""})


def test_agent_result_needs_input_no_resume_context_raises():
    with pytest.raises(ProtocolError, match="resume_context must be non-null"):
        parse_agent_result({"status": "needs_input", "question_for_user": "What?", "resume_context": None})
