import json
import re
from typing import Any, Callable

from voice_transcribe.llm import run_llm_prompt

_TOOL_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def run(
    system_prompt: str,
    initial_message: str,
    tools: list[dict],
    tool_fns: dict[str, Callable[..., Any]],
    llm_backend: str,
    ollama_model: str,
) -> str:
    """Generic agent loop: send context to LLM, parse fenced JSON tool calls, repeat.

    Terminates when the LLM response contains no tool-call block.
    Returns the final LLM response text.
    """
    context = (
        f"{system_prompt}\n\n"
        f"Available tools:\n{json.dumps(tools, indent=2)}\n\n"
        "To call a tool, include exactly one fenced JSON block in your response:\n"
        "```json\n"
        '{"tool": "<name>", "args": {<key>: <value>, ...}}\n'
        "```\n"
        "Omit the block when you are finished.\n\n"
        f"User: {initial_message}\n\n"
        "Assistant:"
    )

    while True:
        response = run_llm_prompt(context, llm_backend, ollama_model)

        match = _TOOL_BLOCK_RE.search(response)
        if not match:
            return response

        try:
            call = json.loads(match.group(1))
            tool_name = call["tool"]
            args = call.get("args", {})
        except (json.JSONDecodeError, KeyError):
            return response

        if tool_name not in tool_fns:
            result = f"Error: unknown tool '{tool_name}'"
        else:
            try:
                result = tool_fns[tool_name](**args)
            except Exception as exc:
                result = f"Error calling {tool_name}: {exc}"

        result_str = json.dumps(result) if not isinstance(result, str) else result
        context = (
            f"{context}\n{response}\n\n"
            f"Tool result ({tool_name}): {result_str}\n\n"
            "Assistant:"
        )
